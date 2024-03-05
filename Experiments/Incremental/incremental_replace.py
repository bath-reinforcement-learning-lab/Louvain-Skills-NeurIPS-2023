import gc
import copy
import uuid
import json
import random

from pathlib import Path
from typing import List

import numpy as np
import igraph as ig
import networkx as nx
import networkx.algorithms.community as nx_comm

from tqdm import tqdm

from simpleoptions import PrimitiveOption, OptionAgent
from simpleoptions.environment import BaseEnvironment
from simpleoptions.option import BaseOption

from louvainskills.louvain import apply_louvain
from louvainskills.options import LouvainOption
from louvainskills.utils.graph_utils import convert_nx_to_ig, convert_ig_to_nx
from louvainskills.envs.discrete_gridworlds import DiscreteRameshMazeBLTR

from incremental_option_trainers import IncrementalLouvainOptionTrainer


class IncrementalLouvainAgent(OptionAgent):
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
    ):
        super().__init__(
            env, test_env, epsilon, macro_alpha, intra_option_alpha, gamma, default_action_value, n_step_updates
        )

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
        Trains the agent for a given number of episodes.

        Args:
            num_epochs (int): The number of epochs to train the agent for.
            epoch_length (int): How many time-steps each epoch should last for.
            process_new_nodes_intervals (List[int]): The time-steps at which the agent should add new nodes to the STG, assign them clusters, and re-define the skill hierarchy.
            render_interval (int, optional): How often (in time-steps) to call the environement's render function, in time-steps. Zero by default, disabling rendering.
            test_interval (int, optional): How often (in epochs) to evaluate the greedy policy learned by the agent. Zero by default, in which case training performance is returned.
            test_length (int, optional): How long (in time-steps) to test the agent for. Zero by default, in which case the agent is tested for one epoch.
            test_runs (int, optional): How many test runs to perform each test_interval.
            verbose_logging (bool, optional): Whether to log all information about each time-step, instead of just rewards. Defaults to True.

        Returns:
            List[float]: A list containing floats representing the rewards earned by the agent each time-step.
        """
        # Set the time-step limit.
        num_time_steps = num_epochs * epoch_length

        # If we are testing the greedy policy separately, make a separate copy of
        # the environment to use for those tests. Also initialise variables for
        # tracking test performance.
        training_rewards = [None for _ in range(num_time_steps)]

        if test_interval > 0:
            test_interval_time_steps = test_interval * epoch_length
            evaluation_rewards = [None for _ in range(num_time_steps // test_interval_time_steps)]

            # Check that a test environment has been provided - if not, raise an error.
            if self.test_env is None:
                raise RuntimeError("No test_env has been provided specified.")
        else:
            evaluation_rewards = []

        # Set the environment's option set to be the set of primitive options.
        options = []
        for action in self.env.get_action_space():
            options.append(PrimitiveOption(action, self.env))
        self.env.set_options(options)
        self.test_env.set_options(options)

        episode = 0
        time_steps = 0

        stg = nx.DiGraph()
        new_nodes = []

        while time_steps < num_time_steps:
            # Initialise initial state variables.
            state = self.env.reset()
            terminal = False

            # If this initial state has not been seen before,
            # add it to the STG and record it as a new node.
            if not stg.has_node(state):
                stg.add_node(state)
                new_nodes.append(state)

            if render_interval > 0:
                self.env.render()
                time_since_last_render = 0

            while not terminal:
                selected_option = self.select_action(state, self.executing_options)

                # Handle if the selected option is a higher-level option.
                if isinstance(selected_option, BaseOption):
                    self.executing_options.append(copy.copy(selected_option))
                    self.executing_options_states.append([state])
                    self.executing_options_rewards.append([])

                # Handle if the selected option is a primitive action.
                else:
                    time_steps += 1
                    next_state, reward, terminal, __ = self.env.step(selected_option)

                    # Logging
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

                    # Render, if we need to.
                    if render_interval > 0 and time_steps % render_interval == 0:
                        self.env.render()

                    # Record newly-seen states and transitions.
                    if not stg.has_node(next_state):
                        stg.add_node(next_state)
                        new_nodes.append(next_state)
                    if not stg.has_edge(state, next_state):
                        stg.add_edge(state, next_state)

                    state = next_state

                    # If this is a time-step when we should be processing new nodes,
                    # process them and update the skill hierarchy.
                    if time_steps in process_new_nodes_intervals:
                        print(f"Time-Step {time_steps}/{num_time_steps}.")
                        if len(new_nodes) > 0:
                            print("Updating STG and Skill Hierarchy...")
                            print(f"{len(new_nodes)} New Nodes Found: {new_nodes}")
                            stg, options = self.update_options(stg)
                            self.env.set_options(options)
                            self.test_env.set_options(options)

                            self.purge_old_q_table()
                            new_nodes = []
                            print("Updated STG and Skill Hierarchy!")

                        # gridlayout(stg)
                        # nx.write_gexf(
                        #     stg,
                        #     f"Incremental Agent - FourRooms - {len(stg.nodes)} Nodes - {time_steps} Decision Stages.gexf",
                        #     prettyprint=True,
                        # )

                        # Write graph to file for inspection.
                        # gridlayout(stg)
                        # nx.write_gexf(
                        #     stg,
                        #     f"Incremental Agent - FourRooms - {len(stg.nodes)} Nodes - {time_steps} Decision Stages.gexf",
                        #     prettyprint=True,
                        # )

                    for i in range(len(self.executing_options)):
                        self.executing_options_states[i].append(next_state)
                        self.executing_options_rewards[i].append(reward)

                    # Terminate any options which need terminating this time-step.
                    while self.executing_options and self._roll_termination(self.executing_options[-1], next_state):
                        # Perform a macro-q learning update for the terminating option.
                        self.macro_q_learn(
                            self.executing_options_states[-1],
                            self.executing_options_rewards[-1],
                            self.executing_options[-1],
                            self.n_step_updates,
                        )
                        # Perform an intra-option learning update for the terminating option.
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

                    # If we are testing the greedy policy learned by the agent separately,
                    # and it is time to test it, then test it.
                    if test_interval > 0 and time_steps % test_interval_time_steps == 0:
                        evaluation_rewards[(time_steps - 1) // test_interval_time_steps] = self.test_policy(
                            test_length,
                            test_runs,
                            time_steps // test_interval_time_steps,
                            allow_exploration=False,
                            verbose_logging=verbose_logging,
                        )

                # If we have been training for more than the desired number of time-steps, terminate.
                if time_steps >= num_time_steps:
                    terminal = True

                # Handle if the current state is terminal.
                if terminal:
                    while len(self.executing_options) > 0:
                        # Perform a macro-q learning update for the topmost option.
                        self.macro_q_learn(
                            self.executing_options_states[-1],
                            self.executing_options_rewards[-1],
                            self.executing_options[-1],
                            self.n_step_updates,
                        )
                        # Perform an intra-option learning update for the topmost option.
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

        if verbose_logging:
            training_log = self.training_log
            evaluation_log = self.evaluation_log if self.evaluation_log else None
            return training_log, evaluation_log
        else:
            training_log = [sum(training_rewards[i * epoch_length : (i + 1) * epoch_length]) for i in range(num_epochs)]
            evaluation_log = evaluation_rewards if evaluation_rewards else None
            return training_log, evaluation_log

    def update_options(self, original_stg: nx.DiGraph):
        # Create a fresh copy of the STG, without any node attributes.
        stg = nx.DiGraph()
        stg.add_nodes_from(original_stg.nodes)
        stg.add_edges_from(original_stg.edges)

        # Re-run the Louvain algorithm from scratch.
        stg_ig = convert_nx_to_ig(stg)
        stg_ig, aggs_ig = apply_louvain(stg_ig, resolution=0.05, return_aggregate_graphs=True, first_levels_to_skip=1)
        stg = convert_ig_to_nx(stg_ig)
        aggs = []
        for i, agg_ig in enumerate(aggs_ig):
            agg = convert_ig_to_nx(agg_ig)

            if agg.number_of_nodes() > 1:
                aggs.append(copy.deepcopy(agg))

        # Extract skill hierarchy from the STG.
        skill_hierarchy = []
        for i, agg in enumerate(aggs[1:]):
            skill_hierarchy.append([])
            for u, v in agg.to_undirected().edges():
                if u != v:
                    skill_hierarchy[i].append((i, u, v))
                    skill_hierarchy[i].append((i, v, u))

        # Define primitive actions.
        primitive_options = []
        for action in self.env.get_action_space():
            primitive_options.append(PrimitiveOption(action, self.env))

        # Instantiate training environment.
        training_env = self.env.__class__()
        training_env.reset()

        # Train Louvain options.
        options = []
        option_trainer = IncrementalLouvainOptionTrainer(
            training_env,
            stg,
            n_episodes=10000,
            max_episode_length=200,
            can_leave_initiation_set=False,
        )
        for level, hierarchy_level in tqdm(enumerate(skill_hierarchy), desc="Hierachy Level"):
            options.append([])

            # Set available options to options from the previous level of the hierarchy.
            if level == 0:
                training_env.options = copy.copy(primitive_options)
            else:
                training_env.options = copy.copy(options[level - 1])

            # Train this level of the hierarchy.
            for i, u, v in tqdm(hierarchy_level, desc="Training Skills"):
                # Train skills at this level of the hierarchy.
                options[level].append(
                    LouvainOption(training_env, stg, i, u, v, False, option_trainer.train_option_policy(i, u, v))
                )

        options = [option for level in options for option in level]
        options.extend(primitive_options)

        # Return the updated STG and options.
        return stg, options

    def purge_old_q_table(self):
        primitive_hashes = []
        for action in self.env.get_action_space():
            primitive_hashes.append(hash(PrimitiveOption(action, self.env)))

        keys_to_del = []
        for key in self.q_table.keys():
            state_hash, action_hash = key
            if action_hash not in primitive_hashes:
                keys_to_del.append(key)

        for key in keys_to_del:
            del self.q_table[key]


if __name__ == "__main__":
    num_runs = 40
    for run in range(num_runs):
        try:
            env_name = "Maze"
            output_directory = "./Training Results/Rooms/Incremental Replace/"
            output_directory_training = "./Training Results/Rooms/Incremental Replace Training/"
            experiment_id = random.randrange(10000)
            num_agents = 1

            # Run Macro-Q Learning Agent
            # Initialise our environment.
            env = DiscreteRameshMazeBLTR()
            test_env = DiscreteRameshMazeBLTR()

            # Initialise our agent and train it for 100x100 time-steps.
            agent = IncrementalLouvainAgent(
                env,
                test_env=test_env,
                epsilon=0.1,
                macro_alpha=0.4,
                intra_option_alpha=0.4,
                gamma=1.0,
                n_step_updates=False,
            )

            train_results, test_results = agent.run_agent(
                num_epochs=100,
                epoch_length=750,
                process_new_nodes_intervals=[500, 2000, 5000, 15000, 30000, 60000],
                test_interval=1,
                test_length=100,
                test_runs=5,
                verbose_logging=False,
            )

            gc.collect()

            # Write testing results to output file.
            Path(output_directory).mkdir(parents=True, exist_ok=True)
            with open(f"{output_directory}/{experiment_id}-{uuid.uuid1()}.json", "w", encoding="utf-8") as f:
                json.dump(test_results, f, ensure_ascii=False, indent=4)

            # Write training results to output file.
            Path(output_directory_training).mkdir(parents=True, exist_ok=True)
            with open(f"{output_directory_training}/{experiment_id}-{uuid.uuid1()}.json", "w", encoding="utf-8") as f:
                json.dump(train_results, f, ensure_ascii=False, indent=4)

        except:
            continue
