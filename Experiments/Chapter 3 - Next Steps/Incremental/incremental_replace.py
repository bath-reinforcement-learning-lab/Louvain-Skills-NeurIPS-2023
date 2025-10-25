import gc
import copy
import uuid
import json
import random

from pathlib import Path
from typing import List

import networkx as nx

from tqdm import tqdm

from simpleoptions import PrimitiveOption, OptionAgent
from simpleoptions.environment import BaseEnvironment
from simpleoptions.option import BaseOption

from louvainskills.louvain import apply_louvain
from louvainskills.options import LouvainOption
from louvainskills.utils.graph_utils import convert_nx_to_ig, convert_ig_to_nx
from louvainskills.incremental_option_trainers import IncrementalValueIterationOptionTrainer

from simpleenvs.envs.discrete_rooms import XuFourRooms


class IncrementalReplaceAgent(OptionAgent):
    def __init__(
        self,
        env: "BaseEnvironment",
        test_env: "BaseEnvironment" = None,
        epsilon: float = 0.15,
        macro_alpha: float = 0.2,
        intra_option_alpha: float = 0.2,
        gamma: float = 1.0,
        default_action_value=0.0,
        n_step_updates=False,
        *,
        vi_theta: float = 1e-8,
        vi_gamma: float = 1.0,
        vi_num_rollouts: int | None = None,
        vi_deterministic: bool = True,
    ):
        """
        Creates a new IncrementalReplaceAgent.

        This agent incrementally constructs its internal state transition graph (STG)
        as it explores its environment. At pre-specified update intervals, it discards
        any existing Louvain skill hierarchy and reconstructs a new one entirely from
        scratch based on the updated STG. All existing high-level skills are replaced,
        and their value functions are reset.

        Args:
            env (BaseEnvironment): The environment in which to train the agent.
            test_env (BaseEnvironment, optional): A second environment used for evaluation runs.
            epsilon (float): The probability of selecting a random option (for exploration).
            macro_alpha (float): The learning rate for macro-level Q-learning updates.
            intra_option_alpha (float): The learning rate for intra-option updates.
            gamma (float): The environment discount factor.
            default_action_value (float): The initial value assigned to unseen (state, option) pairs.
            n_step_updates (bool): Whether to use n-step returns when updating the value function.
            vi_theta (float): The convergence threshold used in the value iteration algorithm.
            vi_gamma (float): The discount factor used in value iteration.
            vi_num_rollouts (int | None): The number of rollouts used to approximate option models
                when the environment is stochastic.
            vi_deterministic (bool): Whether the environment is fully deterministic.
        """
        super().__init__(
            env, test_env, epsilon, macro_alpha, intra_option_alpha, gamma, default_action_value, n_step_updates
        )
        self.vi_theta = vi_theta
        self.vi_gamma = vi_gamma
        self.vi_num_rollouts = vi_num_rollouts
        self.vi_deterministic = vi_deterministic

    def run_agent(
        self,
        num_epochs: int,
        epoch_length: int,
        process_new_nodes_intervals: List[int],
        render_interval: int = 0,
        test_interval: int = 0,
        test_length: int = 0,
        test_runs: int = 10,
        verbose_logging: bool = True,
    ) -> List[float]:
        """
        Trains the agent for a given number of epochs.

        Args:
            num_epochs (int): The number of epochs to train the agent for.
            epoch_length (int): The number of decision stages per epoch.
            process_new_nodes_intervals (List[int]): The decision stages at which the agent should
                add newly discovered nodes to the STG, apply the Louvain algorithm, and reconstruct
                its full skill hierarchy.
            render_interval (int, optional): How often to render the environment (in decision stages).
                Defaults to 0, which disables rendering.
            test_interval (int, optional): How often (in epochs) to evaluate the greedy policy learned
                by the agent. Defaults to 0, which disables evaluation.
            test_length (int, optional): The per-episode cut-off (in decision stages) used during episodic evaluation.
            test_runs (int, optional): The number of evaluation episodes to average over when testing.
            verbose_logging (bool, optional): Whether to log all information about each decision stage,
                instead of only rewards. Defaults to True.

        Returns:
            List[float]: A list containing floats representing the rewards earned by the agent at each decision stage.
        """
        # Compute the total number of decision stages to train for.
        num_time_steps = num_epochs * epoch_length

        # Initialise reward logs.
        training_rewards = [None for _ in range(num_time_steps)]

        # If a separate test interval is specified, prepare episodic evaluation runs.
        if test_interval > 0:
            test_interval_time_steps = test_interval * epoch_length
            episodic_evaluation_rewards = [None for _ in range(num_time_steps // test_interval_time_steps)]

            # Check that a separate test environment has been provided.
            if self.test_env is None:
                raise RuntimeError("No test_env has been provided specified.")
        else:
            episodic_evaluation_rewards = []

        # Set the environment's option set to the set of primitive options.
        options = []
        for action in self.env.get_action_space():
            options.append(PrimitiveOption(action, self.env))
        self.env.set_options(options)
        if self.test_env is not None:
            self.test_env.set_options(options)

        episode = 0
        time_steps = 0

        # Initialise an empty state transition graph.
        stg = nx.DiGraph()
        new_nodes = []

        # ──────────────────────────────────────────────────────────────────────────────
        # Training Loop.
        # ──────────────────────────────────────────────────────────────────────────────
        while time_steps < num_time_steps:
            # Reset the environment to obtain an initial state.
            state = self.env.reset()
            terminal = False

            # If this initial state has not been seen before, add it to the STG.
            if not stg.has_node(state):
                stg.add_node(state)
                new_nodes.append(state)

            if render_interval > 0:
                self.env.render()

            # ──────────────────────────────────────────────────────────────────────────
            # Run one episode.
            # ──────────────────────────────────────────────────────────────────────────
            while not terminal:
                # Select an option using the agent's current policy (ε-greedy).
                selected_option = self.select_action(state, self.executing_options)

                # Handle higher-level options.
                if isinstance(selected_option, BaseOption):
                    self.executing_options.append(copy.copy(selected_option))
                    self.executing_options_states.append([state])
                    self.executing_options_rewards.append([])

                # Handle primitive actions.
                else:
                    time_steps += 1
                    next_state, reward, terminal, __ = self.env.step(selected_option)

                    # Record reward information.
                    training_rewards[time_steps - 1] = reward
                    if verbose_logging:
                        transition = {
                            "state": state,
                            "next_state": next_state,
                            "reward": reward,
                            "terminal": terminal,
                            "active_options": [str(option) for option in self.executing_options],
                        }
                        for key, value in transition.items():
                            self.training_log[key].append(value)

                    # Render the environment, if required.
                    if render_interval > 0 and time_steps % render_interval == 0:
                        self.env.render()

                    # Record any newly discovered states and transitions in the STG.
                    if not stg.has_node(next_state):
                        stg.add_node(next_state)
                        new_nodes.append(next_state)
                    if not stg.has_edge(state, next_state):
                        stg.add_edge(state, next_state)

                    state = next_state

                    # ──────────────────────────────────────────────────────────────────────
                    # Update the skill hierarchy at the specified decision stages.
                    # ──────────────────────────────────────────────────────────────────────
                    if time_steps in process_new_nodes_intervals:
                        print(f"Decision Stage {time_steps}/{num_time_steps}.")
                        if len(new_nodes) > 0:
                            print("Updating STG and skill hierarchy...")
                            print(f"{len(new_nodes)} new nodes discovered: {new_nodes}")
                            stg, options = self.update_options(stg)
                            self.env.set_options(options)
                            if self.test_env is not None:
                                self.test_env.set_options(options)

                            # After rebuilding the hierarchy, purge obsolete entries from the Q-table.
                            self.purge_old_q_table()
                            new_nodes = []
                            print("Updated STG and skill hierarchy!")

                    # Record trajectory information for any executing options.
                    for i in range(len(self.executing_options)):
                        self.executing_options_states[i].append(next_state)
                        self.executing_options_rewards[i].append(reward)

                    # Terminate any options whose termination conditions are met.
                    while self.executing_options and self._roll_termination(self.executing_options[-1], next_state):
                        # Perform a macro-level Q-learning update for the terminating option.
                        self.macro_q_learn(
                            self.executing_options_states[-1],
                            self.executing_options_rewards[-1],
                            self.executing_options[-1],
                            self.n_step_updates,
                        )
                        # Perform an intra-option learning update for the same option.
                        self.intra_option_learn(
                            self.executing_options_states[-1],
                            self.executing_options_rewards[-1],
                            self.executing_options[-1],
                            self.executing_options[-2] if len(self.executing_options) > 1 else None,
                            self.n_step_updates,
                        )
                        self.executing_options_states.pop()
                        self.executing_options_rewards.pop()
                        self.executing_options.pop()

                    # Evaluate the greedy policy at specified test intervals using episodic evaluation.
                    if test_interval > 0 and time_steps % test_interval_time_steps == 0:
                        episodic_evaluation_rewards[(time_steps - 1) // test_interval_time_steps] = self.test_policy(
                            test_length=test_length,  # per-episode cut-off.
                            test_runs=test_runs,  # number of episodes.
                            eval_number=time_steps // test_interval_time_steps,
                            allow_exploration=False,
                            verbose_logging=verbose_logging,
                            episodic_eval=True,  # episodic evaluation enabled.
                        )

                # End episode if the total training duration has been reached.
                if time_steps >= num_time_steps:
                    terminal = True

                # Perform final updates for any options that remain active when the episode ends.
                if terminal:
                    while len(self.executing_options) > 0:
                        self.macro_q_learn(
                            self.executing_options_states[-1],
                            self.executing_options_rewards[-1],
                            self.executing_options[-1],
                            self.n_step_updates,
                        )
                        self.intra_option_learn(
                            self.executing_options_states[-1],
                            self.executing_options_rewards[-1],
                            self.executing_options[-1],
                            self.executing_options[-2] if len(self.executing_options) > 1 else None,
                            self.n_step_updates,
                        )
                        self.executing_options_states.pop()
                        self.executing_options_rewards.pop()
                        self.executing_options.pop()

            episode += 1
        gc.collect()

        # ──────────────────────────────────────────────────────────────────────────────
        # Return logs in the chosen format.
        # ──────────────────────────────────────────────────────────────────────────────
        if verbose_logging:
            # Return detailed training logs and the episodic evaluation log from the base class.
            training_log = self.training_log
            evaluation_log = self.episodic_evaluation_log if self.episodic_evaluation_log else None
            return training_log, evaluation_log
        else:
            # Return per-epoch training rewards and the mean episodic evaluation rewards at each evaluation point.
            training_log = [sum(training_rewards[i * epoch_length : (i + 1) * epoch_length]) for i in range(num_epochs)]
            evaluation_log = episodic_evaluation_rewards if episodic_evaluation_rewards else None
            return training_log, evaluation_log

    def update_options(self, original_stg: nx.DiGraph):
        """
        Constructs a new Louvain skill hierarchy by applying the Louvain algorithm
        to the agent's current state transition graph (STG).

        Args:
            original_stg (nx.DiGraph): The current state transition graph.

        Returns:
            Tuple[nx.DiGraph, List[LouvainOption]]: The updated STG and the full list of available options.
        """
        # Create a fresh copy of the STG without node attributes.
        stg = nx.DiGraph()
        stg.add_nodes_from(original_stg.nodes)
        stg.add_edges_from(original_stg.edges)

        # Apply the Louvain algorithm from scratch.
        stg_ig = convert_nx_to_ig(stg)
        stg_ig, aggs_ig = apply_louvain(stg_ig, resolution=1.0, return_aggregate_graphs=True, first_levels_to_skip=1)
        stg = convert_ig_to_nx(stg_ig)
        aggs = []
        for i, agg_ig in enumerate(aggs_ig):
            agg = convert_ig_to_nx(agg_ig)
            if agg.number_of_nodes() > 1:
                aggs.append(copy.deepcopy(agg))

        # Extract the skill hierarchy from the sequence of aggregate graphs.
        skill_hierarchy = []
        for i, agg in enumerate(aggs[1:]):
            skill_hierarchy.append([])
            for u, v in agg.edges():
                if u != v:
                    skill_hierarchy[i].append((i, u, v))

        # Define primitive options.
        primitive_options = []
        for action in self.env.get_action_space():
            primitive_options.append(PrimitiveOption(action, self.env))

        # Instantiate a fresh copy of the environment for option training.
        training_env = self.env.__class__()
        training_env.reset()

        # Train Louvain options using value iteration.
        options = []
        option_trainer = IncrementalValueIterationOptionTrainer(
            training_env,
            stg,
            gamma=self.vi_gamma,
            theta=self.vi_theta,
            num_rollouts=self.vi_num_rollouts,
            deterministic=self.vi_deterministic,
        )

        for level, hierarchy_level in tqdm(enumerate(skill_hierarchy), desc="Hierarchy Level"):
            options.append([])

            # Set available options to those from the previous level of the hierarchy.
            if level == 0:
                training_env.set_options(copy.copy(primitive_options))
            else:
                training_env.set_options(copy.copy(options[level - 1]))

            # Train this level of the hierarchy.
            for i, u, v in tqdm(hierarchy_level, desc="Training Skills"):
                # Instantiate a Louvain option for moving between the two neighbouring clusters.
                option = LouvainOption(
                    stg=stg,
                    hierarchy_level=i,
                    source_cluster=u,
                    target_cluster=v,
                    can_leave_initiation_set=False,
                )

                # Skip options that are not yet meaningful (e.g., empty initiation set).
                if not option.initiation_set:
                    continue

                # Train the option's policy using value iteration.
                policy = option_trainer.train_option_policy(option, can_leave_initiation_set=False)
                if not policy:
                    continue

                option.policy_dict = policy
                options[level].append(option)

        # Flatten the hierarchy and append primitive options.
        options = [option for level in options for option in level]
        options.extend(primitive_options)

        # Return the updated STG and options.
        return stg, options

    def purge_old_q_table(self):
        """
        Removes Q-table entries associated with obsolete options after the skill hierarchy is rebuilt.
        """
        primitive_hashes = []
        for action in self.env.get_action_space():
            primitive_hashes.append(hash(PrimitiveOption(action, self.env)))

        keys_to_del = []
        for key in list(self.q_table.keys()):
            state_hash, action_hash = key
            if action_hash not in primitive_hashes:
                keys_to_del.append(key)

        for key in keys_to_del:
            del self.q_table[key]


if __name__ == "__main__":
    import traceback

    # Target number of successful runs.
    target_successful_runs = 10

    successful_runs = 0
    attempts = 0

    env_name = "XuFourRooms"
    output_directory = "./Training Results/Chapter 3/Incremental/Rooms/Episode/Replace/"
    output_directory_training = "./Training Results/Chapter 3/Incremental/Rooms/Train/Replace/"

    while successful_runs < target_successful_runs:
        attempts += 1
        print(f"\n[Replace] Attempt {attempts} (successful so far: {successful_runs}/{target_successful_runs})...")

        try:
            experiment_id = random.randrange(10000)

            # Initialise the deterministic environment.
            env = XuFourRooms(movement_penalty=-0.01, goal_reward=1.0)
            test_env = XuFourRooms(movement_penalty=-0.01, goal_reward=1.0)

            # Initialise the incremental agent and train it.
            agent = IncrementalReplaceAgent(
                env,
                test_env=test_env,
                epsilon=0.1,
                macro_alpha=0.4,
                intra_option_alpha=0.4,
                gamma=1.0,
                n_step_updates=True,
                vi_theta=1e-5,
                vi_gamma=0.99,
                vi_num_rollouts=1,
                vi_deterministic=True,
            )

            train_results, test_results = agent.run_agent(
                num_epochs=100,
                epoch_length=100,
                process_new_nodes_intervals=[100, 500, 1000, 3000, 5000, 8000],
                test_interval=1,
                test_length=40,
                test_runs=5,
                verbose_logging=False,
            )

            gc.collect()

            # Write results only for successful runs.
            Path(output_directory).mkdir(parents=True, exist_ok=True)
            with open(f"{output_directory}/{experiment_id}-{uuid.uuid1()}.json", "w", encoding="utf-8") as f:
                json.dump(test_results, f, ensure_ascii=False, indent=4)

            Path(output_directory_training).mkdir(parents=True, exist_ok=True)
            with open(f"{output_directory_training}/{experiment_id}-{uuid.uuid1()}.json", "w", encoding="utf-8") as f:
                json.dump(train_results, f, ensure_ascii=False, indent=4)

            successful_runs += 1
            print(f"[Replace] Run {successful_runs} of {target_successful_runs} completed successfully.")

        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break

        except Exception as e:
            print(f"[Replace] Attempt {attempts} failed with an exception. Skipping this run.")
            print(f"Reason: {e}")
            traceback.print_exc()
            gc.collect()
            continue

    print(f"\n[Replace] Finished. Successful runs: {successful_runs}/{target_successful_runs}.")
