import math
import copy
import random
import warnings

import numpy as np
import igraph as ig
import networkx as nx

from itertools import cycle
from collections import defaultdict

from typing import List, Tuple, Dict, Hashable
from numbers import Number

from simpleoptions import BaseEnvironment, BaseOption, PrimitiveOption, PseudoRewardOption
from simpleoptions.utils.math import discounted_return


class OptionTrainer(object):
    def __init__(self):
        pass

    def train_option_policy(self):
        pass


class ValueIterationOptionTrainer(OptionTrainer):
    def __init__(
        self,
        env: BaseEnvironment,
        stg: nx.Graph,
        gamma: float,
        theta: float,
        num_rollouts: int = None,
        deterministic: bool = False,
    ):
        """
        Initialises a new ValueIterationOptionTrainer.

        Args:
            env (BaseEnvironment): The environment in which the options are to be trained.
            gamma (float): The discount factor used in the value iteration algorithm.
            theta (float): The convergence threshold used in the value iteration algorithm.
            num_rollouts (int, optional): The number of rollouts used to approximate option models. Defaults to None.
            deterministic (bool, optional): Whether the environment and options are fully deterministic. Defaults to False.

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

    def train_option_policy(
        self,
        option_to_train: PseudoRewardOption,
        can_leave_initiation_set: bool,
    ) -> PseudoRewardOption:

        bottom_level = all([isinstance(option, PrimitiveOption) for option in self.env.get_option_space()])

        if bottom_level:
            option_to_train.policy_dict = self._primitive_value_iteration(option_to_train, can_leave_initiation_set)
        else:
            option_to_train.policy_dict = self._hierarchical_value_iteration(option_to_train, can_leave_initiation_set)

        return option_to_train

    def _primitive_value_iteration(
        self,
        option_to_train: PseudoRewardOption,
        can_leave_initiation_set: bool,
    ) -> Dict[Hashable, PrimitiveOption]:

        # Define set of states to learn a policy in.
        if not can_leave_initiation_set:
            option_to_train.initiation_set = {
                state for state in option_to_train.initiation_set if not self.env.is_state_terminal(state)
            }
            state_set = option_to_train.initiation_set

        else:
            option_to_train.executable_set = {
                state for state in option_to_train.executable_set if not self.env.is_state_terminal(state)
            }
            state_set = option_to_train.executable_set

        # Initialise state value function to zero.
        state_values = {state: 0 for state in self.env.get_state_space()}

        # Loop until the state value function converges upon the optimal state value function.
        while True:
            delta = 0
            for state in state_set:

                v_curr = state_values[state]

                next_state_values = []
                for action in self.env.get_available_actions(state):
                    successors = self.env.get_successors(state, [action])
                    next_state_values.append(
                        sum(
                            [
                                trans_prob
                                * (
                                    option_to_train.pseudo_reward(state, action, next_state)
                                    + self.gamma * state_values[next_state]
                                )
                                for (next_state, _), trans_prob in successors
                            ]
                        )
                    )
                state_values[state] = max(next_state_values)

                delta = max(delta, abs(v_curr - state_values[state]))

            if delta < self.theta:
                break

        # Output the optimal policy by acting greedily with respect to the learned state value function.
        policy = {}
        for state in state_set:

            action_values = {}
            for action in self.env.get_available_actions(state):
                successors = self.env.get_successors(state, [action])
                action_values[action] = sum(
                    [
                        trans_prob
                        * (
                            option_to_train.pseudo_reward(state, action, next_state)
                            + self.gamma * state_values[next_state]
                        )
                        for (next_state, _), trans_prob in successors
                    ]
                )
            policy[state] = max(action_values, key=action_values.get)

        # Convert policy over primtive actions into a policy over primitive options.
        primitive_options = {
            option.action: option for option in self.env.get_option_space() if isinstance(option, PrimitiveOption)
        }
        policy = {state: primitive_options[action] for state, action in policy.items()}

        # For Debugging - adds policy labels to the STG.
        for state in state_set:
            if not ((state is None) or (self.env.is_state_terminal(state))):
                self.stg.nodes[state][f"{str(option_to_train)}"] = str(policy[state])

        return policy

    def _hierarchical_value_iteration(
        self, option_to_train: PseudoRewardOption, can_leave_initiation_set: bool
    ) -> Dict[Hashable, BaseOption]:
        # Define set of states to learn a policy in.
        if not can_leave_initiation_set:
            option_to_train.initiation_set = {
                state for state in option_to_train.initiation_set if not self.env.is_state_terminal(state)
            }
            state_set = option_to_train.initiation_set

        else:
            option_to_train.executable_set = {
                state for state in option_to_train.executable_set if not self.env.is_state_terminal(state)
            }
            state_set = option_to_train.executable_set

        ######################################################################
        ### STEP 1: Roll-out options in each state to learn option models. ###
        ######################################################################

        # For each state in state_set, learn a model for each option from the previous level of the hierarchy.
        option_models: Dict[BaseOption, OptionModel] = {}
        for initiating_state in state_set:
            for option in self.env.get_available_options(initiating_state):
                for _ in range(self.num_rollouts):
                    # Define a new model for the option if it hasn't been seen before.
                    if option not in option_models:
                        option_models[option] = OptionModel()

                    # Initialise the environment in the given state.
                    state = self.env.reset(initiating_state)
                    rewards = []
                    done = False
                    k = 0

                    # Execute the option until either it terminates, the option being trained terminates, or a terminal environment state is reached.
                    # As you're going along, keep track of the states visited and the rewards earned.
                    while not done and not option.termination(state) and not option_to_train.termination(state):
                        action = self._get_primitive_option(state, option.policy(state)).policy(state)
                        next_state, _, done, _ = self.env.step(action)
                        reward = option_to_train.pseudo_reward(state, action, next_state)
                        rewards.append(reward)
                        state = next_state
                        k += 1

                    terminating_state = state
                    option_models[option].update(
                        initiating_state, terminating_state, discounted_return(rewards, self.gamma), k
                    )

        ##########################################################
        # Step 2: Compute the transition models for each option. #
        ##########################################################
        for option in option_models:
            option_models[option].compute_model()

        #################################################################################################
        # Step 3: Use the models of lower-level options to learn an policy for the higher-level option. #
        #################################################################################################

        # Initialise state value function to zero.
        state_values = {state: 0 for state in self.env.get_state_space()}

        # Loop until the state value function converges upon the optimal state value function.
        while True:
            delta = 0
            for state in state_set:

                v_curr = state_values[state]

                next_state_values = []
                for option in self.env.get_available_options(state):
                    successors = option_models[option].possible_outcomes(state)
                    next_state_values.append(
                        sum(
                            [
                                trans_prob * (disc_return + self.gamma**k * state_values[next_state])
                                for (next_state, disc_return, k), trans_prob in successors
                            ]
                        )
                    )
                state_values[state] = max(next_state_values)

                delta = max(delta, abs(v_curr - state_values[state]))

            if delta < self.theta:
                break

        # Output the optimal policy by acting greedily with respect to the learned state value function.
        policy: Dict[Hashable, BaseOption] = {}
        for state in state_set:
            option_values = {}
            for option in self.env.get_available_options(state):
                successors = option_models[option].possible_outcomes(state)
                option_values[option] = sum(
                    [
                        trans_prob * (disc_return + self.gamma**k * state_values[next_state])
                        for (next_state, disc_return, k), trans_prob in successors
                    ]
                )
            policy[state] = max(option_values, key=option_values.get)

        # For Debugging - adds policy labels to the STG.
        for state in state_set:
            if not ((state is None) or (self.env.is_state_terminal(state))):
                self.stg.nodes[state][f"{str(option_to_train)}"] = str(policy[state])

        return policy

    def _get_primitive_option(self, state: Hashable, option: BaseOption) -> PrimitiveOption:
        # Recursively query the option policy until we get a primitive option.
        if isinstance(option, PrimitiveOption):
            return option
        else:
            return self._get_primitive_option(state, option.policy(state))


class OptionModel(object):
    def __init__(self):
        self.observed_transitions = []
        self.transition_model = {}
        self.trained = False
        self.up_to_date = True

    def update(self, initiating_state: Hashable, terminating_state: Hashable, disc_return: float, execution_time: int):
        """
        Adds a new transition to the set of experiences used to train the model.

        Args:
            initiating_state (Hashable): The state in which the option began executing.
            terminating_state (Hashable): The state in which the option stopped executing.
            disc_return (float): The discounted return earned by the agent while executing the option.
            execution_time (int): The number of time steps taken to execute the option.
        """
        self.observed_transitions.append((initiating_state, terminating_state, disc_return, execution_time))
        self.up_to_date = False

    def compute_model(self):
        """
        Computes the transition model from the set of observed transitions added using `update`.
        """
        # For each initiating state, compute the number of times transitions to each unique (terminating_state, discounted_return) pair occurred.
        for initiating_state, terminating_state, discounted_return, k in self.observed_transitions:
            if initiating_state not in self.transition_model:
                self.transition_model[initiating_state] = {}
            if (terminating_state, discounted_return, k) not in self.transition_model[initiating_state]:
                self.transition_model[initiating_state][(terminating_state, discounted_return, k)] = 0
            self.transition_model[initiating_state][(terminating_state, discounted_return, k)] += 1

        # Now, for each initiating state, convert the counts into probabilities.
        for initiating_state in self.transition_model:
            total_transitions = sum(self.transition_model[initiating_state].values())
            for successor in self.transition_model[initiating_state]:
                self.transition_model[initiating_state][successor] /= float(total_transitions)

        self.trained = True
        self.up_to_date = True

    def sample(self, initiating_state: Hashable, deterministic: bool = False) -> Tuple[Hashable, float]:
        """
        Samples a terminating state and discounted return for the given initiating state.

        Args:
            initiating_state (Hashable): The state in which the option began executing.
            deterministic (bool, optional): Whether to sample the most likely terminating state deterministically. Defaults to False.

        Raises:
            RuntimeError: Rasied if the model has not been trained yet.

        Returns:
            Tuple[Hashable, float]: The sampled terminating state and discounted return.
        """

        if not self.trained:
            raise RuntimeError(
                "Model has not been trained yet! Add some transitions using `update`, then call `compute_model` first."
            )
        # Using the learned model, sample a terminating state and discounted return for the given initiating state.
        possible_outcomes = self.possible_outcomes(initiating_state)
        if deterministic:
            return max(possible_outcomes, key=lambda x: x[1])[0]
        else:
            return random.choices(possible_outcomes, weights=[prob for (_, prob) in possible_outcomes])[0][0]

    def possible_outcomes(self, initiating_state: Hashable) -> List[Tuple[Tuple[Hashable, float, int], float]]:
        """
        Returns a list of possible terminating states and discounted returns for the given initiating state and their probabilities of occurring.

        Args:
            initiating_state (Hashable): The state in which the option began executing.

        Returns:
            List[Tuple[Tuple[Hashable, float, int], float]]: A list of possible terminating states and discounted returns and their probabilities of occurring.
        """
        return self.transition_model[initiating_state].items()
