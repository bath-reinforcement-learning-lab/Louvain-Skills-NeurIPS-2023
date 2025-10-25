import uuid
import warnings
from collections import deque
from typing import Dict, Hashable

import networkx as nx

from simpleoptions import BaseEnvironment, BaseOption, PrimitiveOption, PseudoRewardOption
from simpleoptions.utils.math import discounted_return

from louvainskills.option_trainers import OptionTrainer, OptionModel
from louvainskills.utils.graph_layouts import gridlayout


class IncrementalValueIterationOptionTrainer(OptionTrainer):
    def __init__(
        self,
        env: BaseEnvironment,
        stg: nx.DiGraph,
        gamma: float,
        theta: float,
        num_rollouts: int = None,
        deterministic: bool = False,
        max_rollout_steps: int = 5000,
    ):
        """
        Initialises a new IncrementalValueIterationOptionTrainer.

        Args:
            env (BaseEnvironment): The environment in which the options are to be trained.
            stg (nx.DiGraph): The (partially discovered) state transition graph.
            gamma (float): The discount factor used in the value iteration algorithm.
            theta (float): The convergence threshold used in the value iteration algorithm.
            num_rollouts (int, optional): The number of rollouts used to approximate option models when deterministic is False.
            deterministic (bool, optional): Whether the environment and options are fully deterministic. Defaults to False.
            max_rollout_steps (int, optional): A hard cap on per-rollout steps in hierarchical rollouts. Defaults to 5000.

        Raises:
            ValueError: Raised if num_rollouts is not provided when deterministic is False.
        """
        self.env = env
        self.stg = stg
        self.theta = theta
        self.gamma = gamma

        if num_rollouts is None and not deterministic:
            raise ValueError("num_rollouts must be provided if deterministic is False.")
        if deterministic and num_rollouts is not None:
            warnings.warn("num_rollouts specified but deterministic is True. num_rollouts will be ignored.")

        self.num_rollouts = 1 if deterministic else num_rollouts
        self.deterministic = deterministic
        self.max_rollout_steps = max_rollout_steps

        # Defaults used by hierarchical rollout safety checks (can be overridden by attribute injection if needed).
        self.max_rollout_state_visits = 64
        self.max_no_progress_steps = 256

    def train_option_policy(
        self,
        option_to_train: PseudoRewardOption,
        can_leave_initiation_set: bool,
    ) -> Dict[Hashable, BaseOption]:
        """
        Trains a policy for the provided option using value iteration, restricted to the
        currently discovered portion of the STG.

        Args:
            option_to_train (PseudoRewardOption): The option whose policy is to be trained.
            can_leave_initiation_set (bool): Whether the option may leave its initiation set.

        Returns:
            Dict[Hashable, BaseOption]: A mapping from state to chosen BaseOption/PrimitiveOption.
                                       Returns {} if no valid training domain exists yet.
        """
        bottom_level = all(isinstance(option, PrimitiveOption) for option in self.env.get_option_space())

        if bottom_level:
            return self._primitive_value_iteration_restricted(option_to_train, can_leave_initiation_set)
        else:
            return self._hierarchical_value_iteration_restricted(option_to_train, can_leave_initiation_set)

    def _primitive_value_iteration_restricted(
        self,
        option_to_train: PseudoRewardOption,
        can_leave_initiation_set: bool,
    ) -> Dict[Hashable, PrimitiveOption]:
        # Define set of states to learn a policy in.
        if not can_leave_initiation_set:
            option_to_train.initiation_set = {
                s for s in option_to_train.initiation_set if (s in self.stg) and (not self.env.is_state_terminal(s))
            }
            state_set = option_to_train.initiation_set
        else:
            option_to_train.executable_set = {
                s for s in option_to_train.executable_set if (s in self.stg) and (not self.env.is_state_terminal(s))
            }
            state_set = option_to_train.executable_set

        if not state_set:
            return {}

        # Initialise state value function to zero over discovered states only.
        state_values = {state: 0 for state in self.env.get_state_space() if state in self.stg}

        # Loop until convergence.
        while True:
            delta = 0
            for state in state_set:
                v_curr = state_values.get(state, 0)

                next_state_values = []
                actions = self.env.get_available_actions(state)
                if not actions:
                    continue

                for action in actions:
                    successors = self.env.get_successors(state, [action])

                    def term(next_state, trans_prob):
                        # Reward: safe (penalises unknown edges/nodes/labels or terminals).
                        r = self._safe_pseudo_reward(option_to_train, state, action, next_state)
                        # Bootstrap only if the edge (state -> next_state) is known and next_state is non-terminal.
                        cont = (
                            self.gamma * state_values.get(next_state, 0)
                            if self.stg.has_edge(state, next_state) and not self.env.is_state_terminal(next_state)
                            else 0.0
                        )
                        return trans_prob * (r + cont)

                    next_state_values.append(
                        sum(term(next_state, trans_prob) for (next_state, _), trans_prob in successors)
                    )

                if next_state_values:
                    state_values[state] = max(next_state_values)

                delta = max(delta, abs(v_curr - state_values.get(state, 0)))

            if delta < self.theta:
                break

        # Output greedy policy.
        policy = {}
        for state in state_set:
            actions = self.env.get_available_actions(state)
            if not actions:
                continue

            action_values = {}
            for action in actions:
                successors = self.env.get_successors(state, [action])

                def q_term(next_state, trans_prob):
                    r = self._safe_pseudo_reward(option_to_train, state, action, next_state)
                    cont = (
                        self.gamma * state_values.get(next_state, 0)
                        if self.stg.has_edge(state, next_state) and not self.env.is_state_terminal(next_state)
                        else 0.0
                    )
                    return trans_prob * (r + cont)

                action_values[action] = sum(
                    q_term(next_state, trans_prob) for (next_state, _), trans_prob in successors
                )

            if action_values:
                best_action = max(action_values, key=action_values.get)
                policy[state] = best_action

        # Convert to primitive options.
        primitive_options = {
            option.action: option for option in self.env.get_option_space() if isinstance(option, PrimitiveOption)
        }
        policy = {state: primitive_options[action] for state, action in policy.items() if action in primitive_options}

        # For debugging – add policy labels to the STG.
        for state in state_set:
            if not ((state is None) or (self.env.is_state_terminal(state))):
                if state in policy:
                    self.stg.nodes[state][f"{str(option_to_train)}"] = str(policy[state])

        return policy

    def _hierarchical_value_iteration_restricted(
        self,
        option_to_train: PseudoRewardOption,
        can_leave_initiation_set: bool,
    ) -> Dict[Hashable, BaseOption]:
        # Define set of states to learn a policy in.
        if not can_leave_initiation_set:
            option_to_train.initiation_set = {
                s for s in option_to_train.initiation_set if (s in self.stg) and (not self.env.is_state_terminal(s))
            }
            state_set = option_to_train.initiation_set
        else:
            option_to_train.executable_set = {
                s for s in option_to_train.executable_set if (s in self.stg) and (not self.env.is_state_terminal(s))
            }
            state_set = option_to_train.executable_set

        if not state_set:
            return {}

        ######################################################################
        ### STEP 1: Roll-out options in each state to learn option models. ###
        ######################################################################

        option_models: Dict[BaseOption, OptionModel] = {}

        max_steps = getattr(self, "max_rollout_steps", 5000)
        max_visits = getattr(self, "max_rollout_state_visits", 64)
        max_no_progress = getattr(self, "max_no_progress_steps", 256)

        for initiating_state in state_set:
            available_options = self.env.get_available_options(initiating_state)
            if not available_options:
                continue

            for option in available_options:
                for _ in range(self.num_rollouts):
                    if option not in option_models:
                        option_models[option] = OptionModel()

                    state = self.env.reset(initiating_state)
                    rewards = []
                    done = False
                    k = 0
                    visits: Dict[Hashable, int] = {}
                    no_progress = 0
                    last_states = deque(maxlen=5)

                    while (
                        (not done)
                        and (state in self.stg)
                        and (not option.termination(state))
                        and (not option_to_train.termination(state))
                    ):
                        last_states.append(state)

                        # Cycle detection: cap repeated visits to the same state.
                        visits[state] = visits.get(state, 0) + 1
                        if visits[state] >= max_visits:
                            fname = f"Incremental Trainer - Rollout State-Visit Loop - {len(self.stg.nodes)} Nodes - {uuid.uuid1()}.gexf"
                            gridlayout(self.stg)
                            nx.write_gexf(self.stg, fname)
                            raise RuntimeError(
                                f"Hierarchical rollout detected a pathological loop (state revisit cap). "
                                f"Option: {option}; initiating_state: {initiating_state}; "
                                f"last_states: {list(last_states)}; graph saved to: {fname}"
                            )

                        primitive = self._get_primitive_option(state, option.policy(state))
                        action = primitive.policy(state)
                        next_state, _, done, _ = self.env.step(action)

                        # No-progress detector (e.g., bumping a wall repeatedly).
                        if next_state == state:
                            no_progress += 1
                            if no_progress >= max_no_progress:
                                fname = f"Incremental Trainer - Rollout No-Progress - {len(self.stg.nodes)} Nodes - {uuid.uuid1()}.gexf"
                                gridlayout(self.stg)
                                nx.write_gexf(self.stg, fname)
                                raise RuntimeError(
                                    f"Hierarchical rollout detected no-progress behaviour. "
                                    f"Option: {option}; initiating_state: {initiating_state}; "
                                    f"last_states: {list(last_states)}; graph saved to: {fname}"
                                )
                        else:
                            no_progress = 0

                        # Truncate if we step outside the known STG footprint.
                        if next_state not in self.stg or not self.stg.has_edge(state, next_state):
                            rewards.append(-1.0)
                            break

                        # Treat transitions into terminal states as failure.
                        if self.env.is_state_terminal(next_state):
                            rewards.append(-1.0)
                            done = True
                            break

                        reward = option_to_train.pseudo_reward(state, action, next_state)
                        rewards.append(reward)
                        state = next_state
                        k += 1

                        if k >= max_steps:
                            fname = f"Incremental Trainer - Rollout Timeout - {len(self.stg.nodes)} Nodes - {uuid.uuid1()}.gexf"
                            gridlayout(self.stg)
                            nx.write_gexf(self.stg, fname)
                            raise RuntimeError(
                                f"Hierarchical rollout exceeded max steps ({max_steps}). "
                                f"Option: {option}; initiating_state: {initiating_state}; steps={k}; "
                                f"last_states: {list(last_states)}; graph saved to: {fname}"
                            )

                    terminating_state = state if state in self.stg else initiating_state
                    option_models[option].update(
                        initiating_state, terminating_state, discounted_return(rewards, self.gamma), k
                    )

        ##########################################################
        # Step 2: Compute the transition models for each option. #
        ##########################################################
        for option in option_models:
            option_models[option].compute_model()

        #################################################################################################
        # Step 3: Use the models of lower-level options to learn a policy for the higher-level option.  #
        #################################################################################################

        state_values = {state: 0 for state in self.env.get_state_space() if state in self.stg}

        while True:
            delta = 0
            for state in state_set:
                v_curr = state_values.get(state, 0)

                # Only consider options that have outcomes for THIS state.
                options_here = [
                    o
                    for o in self.env.get_available_options(state)
                    if (o in option_models) and (state in option_models[o].transition_model)
                ]
                if not options_here:
                    delta = max(delta, abs(v_curr - state_values.get(state, 0)))
                    continue

                option_values = []
                for option in options_here:
                    successors = option_models[option].possible_outcomes(state)
                    option_values.append(
                        sum(
                            trans_prob * (disc_return + self.gamma**k * state_values.get(next_state, 0))
                            for (next_state, disc_return, k), trans_prob in successors
                        )
                    )

                state_values[state] = max(option_values)
                delta = max(delta, abs(v_curr - state_values[state]))

            if delta < self.theta:
                break

        # Output greedy policy.
        policy: Dict[Hashable, BaseOption] = {}
        for state in state_set:
            options_here = [
                o
                for o in self.env.get_available_options(state)
                if (o in option_models) and (state in option_models[o].transition_model)
            ]
            if not options_here:
                continue

            option_values = {}
            for option in options_here:
                successors = option_models[option].possible_outcomes(state)
                option_values[option] = sum(
                    trans_prob * (disc_return + self.gamma**k * state_values.get(next_state, 0))
                    for (next_state, disc_return, k), trans_prob in successors
                )

            policy[state] = max(option_values, key=option_values.get)

        # For debugging – add policy labels to the STG.
        for state in state_set:
            if not ((state is None) or (self.env.is_state_terminal(state))):
                if state in policy:
                    self.stg.nodes[state][f"{str(option_to_train)}"] = str(policy[state])

        return policy

    def _get_primitive_option(self, state: Hashable, option: BaseOption) -> PrimitiveOption:
        if isinstance(option, PrimitiveOption):
            return option
        else:
            return self._get_primitive_option(state, option.policy(state))

    def _safe_pseudo_reward(self, option: PseudoRewardOption, state, action, next_state) -> float:
        """
        Returns a pseudo-reward that is safe when `next_state` or the edge (state -> next_state)
        is not yet in the discovered STG, when `next_state` lacks the required cluster attribute,
        or when `next_state` is terminal.
        """
        if self.env.is_state_terminal(next_state):
            return -1.0

        # Unknown node ⇒ outside known STG.
        if next_state not in self.stg:
            return -1.0

        # Unknown edge ⇒ outside known STG.
        if not self.stg.has_edge(state, next_state):
            return -1.0

        # Missing cluster label at this hierarchy level ⇒ treat as outside known graph for this option.
        cluster_key = f"cluster-{option.hierarchy_level}"
        if cluster_key not in self.stg.nodes[next_state]:
            return -1.0

        # Otherwise, delegate to the option's true pseudo-reward.
        return option.pseudo_reward(state, action, next_state)
