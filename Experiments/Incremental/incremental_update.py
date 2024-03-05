import gc
import math
import uuid
import json
import random
import operator
import itertools

from pathlib import Path
from typing import List

import numpy as np
import igraph as ig
import networkx as nx
import networkx.algorithms.community as nx_comm

from tqdm import tqdm
from copy import copy

from simpleoptions import PrimitiveOption, OptionAgent
from simpleoptions.environment import BaseEnvironment
from simpleoptions.option import BaseOption

from louvainskills.options import LouvainOption
from louvainskills.utils.graph_utils import get_all_neighbours
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
                    self.executing_options.append(copy(selected_option))
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
                            print("Updating STG...")
                            stg = self.update_stg(stg, new_nodes)
                            print(new_nodes)
                            new_nodes = []
                            print("Updated STG!")

                        # gridlayout(stg)
                        # nx.write_gexf(
                        #     stg,
                        #     f"Incremental Agent - FourRooms - {len(stg.nodes)} Nodes - {time_steps} Decision Stages.gexf",
                        #     prettyprint=True,
                        # )

                        print("Training Option Hierarchy...")
                        options = self.update_options(stg)
                        self.env.set_options(options)
                        self.test_env.set_options(options)
                        print("Trained Option Hierarchy!\n")

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

    def update_stg(self, stg, new_nodes):
        # First, assign new nodes to clusters.
        stg = self.assign_node_clusters(stg, new_nodes)

        # Next, deal with disconnected clusters which occur as a result of the Louvain algorithm.
        stg = self._reassign_disconnected_clusters(stg)

        # Return the updated STG.
        return stg

    def update_options(self, stg: nx.DiGraph):
        # Get the set of primitive options for this environment.
        primitive_options = []
        for action in self.env.get_action_space():
            primitive_options.append(PrimitiveOption(action, self.env))

        # Get the number of levels in the hierarchy.
        num_levels = self._get_current_number_of_levels(stg)

        # Create a list of higher-level skills to train.
        skill_hierarchy = []
        for i in range(num_levels):
            skill_hierarchy.append(set())
            # Look at each node in each cluster at this level of the hierarchy, and look at the
            # other clusters that it is possible to transition to.
            clusters = set(nx.get_node_attributes(stg, f"cluster-{i}").values())
            for cluster in clusters:
                nodes_in_cluster = [node for node in stg.nodes if stg.nodes[node][f"cluster-{i}"] == cluster]
                for node in nodes_in_cluster:
                    neighbours = get_all_neighbours(stg, node)
                    for neighbour in neighbours:
                        if stg.nodes[neighbour][f"cluster-{i}"] != cluster:
                            # skill_hierarchy[i].append((i, u, v))
                            skill_hierarchy[i].add(
                                (i, stg.nodes[node][f"cluster-{i}"], stg.nodes[neighbour][f"cluster-{i}"])
                            )

        # Train higher-level options.
        training_env = self.env.__class__()
        training_env.reset()
        options = []
        option_trainer = IncrementalLouvainOptionTrainer(
            # training_env, stg, n_episodes=10000, max_episode_length=200, can_leave_initiation_set=False
            training_env,
            stg,
            n_episodes=5000,
            max_episode_length=200,
            can_leave_initiation_set=False,
        )

        for level, hierarchy_level in tqdm(enumerate(skill_hierarchy), desc="Hierachy Level"):
            options.append([])

            # Set available options to options from the previous level of the hierarchy.
            if level == 0:
                training_env.options = copy(primitive_options)
            else:
                training_env.options = copy(options[level - 1])

            # Train this level of the hierarchy.
            for i, u, v in tqdm(hierarchy_level, desc="Training Skills"):
                # Train skills at this level of the hierarchy.
                options[level].append(
                    LouvainOption(training_env, stg, i, u, v, option_trainer.train_option_policy(i, u, v))
                )

            # gridlayout(stg)
            # nx.write_gexf(
            #     stg,
            #     f"Incremental Agent - FourRooms - {len(stg.nodes)} Nodes - Level {level} Trained.gexf",
            #     prettyprint=True,
            # )

        # Finally, flatten list of higher-level options and add the primitive options, then return them.
        options = [option for level in options for option in level]
        options.extend(primitive_options)
        # print(options)
        return options

    def assign_node_clusters(self, original_stg: nx.DiGraph, new_nodes):
        stg = original_stg.copy()
        existing_clusters = self._get_existing_clusters(stg, 0)
        current_levels = self._get_current_number_of_levels(stg)

        # Add each new node to its own cluster.
        new_cluster_id = max(existing_clusters, default=-1) + 1
        for node in new_nodes:
            stg.nodes[node]["cluster-0"] = new_cluster_id
            new_cluster_id += 1

        # Iterate through each new node and place it in the neighbouring cluster that maximises modularity.
        # Repeat until no increase in modularity can be found.
        while True:
            last_modularity = nx_comm.modularity(stg, self._get_clusters_from_level(stg, 0), weight=None)
            for node in new_nodes:
                # Get neighbouring nodes and their respective clusters.
                neighbours = get_all_neighbours(stg, node)
                neighbour_clusters = set(
                    {node: nx.get_node_attributes(stg, "cluster-0")[node] for node in neighbours}.values()
                )
                own_cluster = stg.nodes[node]["cluster-0"]

                # Assign node to the cluster that maximises modularity.
                best_modularity = nx_comm.modularity(stg, self._get_clusters_from_level(stg, 0), weight=None)
                best_cluster = own_cluster
                for cluster in neighbour_clusters:
                    stg.nodes[node]["cluster-0"] = cluster

                    modularity = nx_comm.modularity(stg, self._get_clusters_from_level(stg, 0), weight=None)

                    if modularity >= best_modularity:
                        best_modularity = modularity
                        best_cluster = cluster

                stg.nodes[node]["cluster-0"] = best_cluster

            if math.isclose(best_modularity, last_modularity, abs_tol=1e-6):
                break

        # Assign new nodes to appropriate higher-level clusters.
        nodes_to_merge = []
        for node in new_nodes:
            # If this node has been assigned to an existing cluster, we can derive its higher-level
            # cluster membership from other nodes in its cluster.
            if stg.nodes[node]["cluster-0"] in existing_clusters:
                # Get another node in this cluster, so that we can copy its cluster membership.
                node_from_same_cluster = next(
                    neighbour
                    for neighbour in stg.nodes
                    if stg.nodes[neighbour]["cluster-0"] == stg.nodes[node]["cluster-0"] and neighbour not in new_nodes
                )

                for att in stg.nodes[node_from_same_cluster]:
                    if att.startswith("cluster-"):
                        stg.nodes[node][att] = stg.nodes[node_from_same_cluster][att]
            # Otherwise, we set it aside for adding to a new higher-level cluster.
            else:
                nodes_to_merge.append(node)

        if current_levels <= 1:
            return self.assign_new_higher_level_clusters(stg, list(stg.nodes), 1, current_levels)
        else:
            if len(nodes_to_merge) == 0:
                return stg
            else:
                return self.assign_new_higher_level_clusters(stg, nodes_to_merge, 1, current_levels)

    def assign_new_higher_level_clusters(self, original_stg: nx.DiGraph, new_nodes, level, num_existing_levels):
        stg = original_stg.copy()
        existing_clusters = self._get_existing_clusters(stg, level)

        # Add each new lower-level cluster to its own higher-level cluster.
        clusters_processed = []
        lower_level_clusters = []
        new_cluster_id = max(existing_clusters, default=-1) + 1
        nodes_labelled = 0
        for new_node in new_nodes:
            if stg.nodes[new_node][f"cluster-{level-1}"] in clusters_processed:
                continue

            # Get all of the nodes in this node's level - 1 cluster.
            nodes_in_same_cluster = [
                node
                for node in stg.nodes
                if stg.nodes[new_node][f"cluster-{level-1}"] == stg.nodes[node][f"cluster-{level-1}"]
            ]
            lower_level_clusters.append(nodes_in_same_cluster)

            # Add the new nodes to their own cluster.
            for node in nodes_in_same_cluster:
                stg.nodes[node][f"cluster-{level}"] = new_cluster_id
                nodes_labelled += 1
            new_cluster_id += 1

            clusters_processed.append(stg.nodes[new_node][f"cluster-{level-1}"])

        # Iterate through each new lower-level cluster and place it in the neighbouring super-cluster
        # that maximises modularity. Repeat until no increase in modularity can be found.
        while True:
            try:
                last_modularity = nx_comm.modularity(stg, self._get_clusters_from_level(stg, level), weight=None)
            except:
                print("Exited early.")
                quit()

            for cluster_of_nodes in lower_level_clusters:
                # Get the neighbours of all of the nodes in this cluster and their respective clusters.
                neighbours = []
                for node in cluster_of_nodes:
                    for neighbour in get_all_neighbours(stg, node):
                        if neighbour not in cluster_of_nodes:
                            neighbours.append(neighbour)
                neighbour_clusters = set(
                    {node: nx.get_node_attributes(stg, f"cluster-{level}")[node] for node in neighbours}.values()
                )
                own_cluster = stg.nodes[cluster_of_nodes[0]][f"cluster-{level}"]

                # Assign node to the cluster that maximises modularity.
                best_modularity = nx_comm.modularity(stg, self._get_clusters_from_level(stg, level), weight=None)
                best_cluster = own_cluster
                for cluster in neighbour_clusters:
                    for node in cluster_of_nodes:
                        stg.nodes[node][f"cluster-{level}"] = cluster

                    modularity = nx_comm.modularity(stg, self._get_clusters_from_level(stg, level), weight=None)

                    if modularity >= best_modularity:
                        best_modularity = modularity
                        best_cluster = cluster

                for node in cluster_of_nodes:
                    stg.nodes[node][f"cluster-{level}"] = best_cluster

            if math.isclose(best_modularity, last_modularity, abs_tol=1e-6):
                break

        # Deal with clusters that have been merged with existing clusters.
        # Send other clusters to be merged in the next level of the hierarchy.
        nodes_to_merge = []
        for cluster_of_nodes in lower_level_clusters:
            # If this cluster has been merged with an existing cluster, we can derive its higher-level
            # cluster membership from other nodes in its cluster.
            if stg.nodes[cluster_of_nodes[0]][f"cluster-{level}"] in existing_clusters:
                # Get another node in this cluster, so that we can copy its cluster membership.
                node_from_same_cluster = next(
                    neighbour
                    for neighbour in stg.nodes
                    if stg.nodes[neighbour][f"cluster-{level}"] == stg.nodes[cluster_of_nodes[0]][f"cluster-{level}"]
                    and neighbour not in new_nodes
                )
                for node in cluster_of_nodes:
                    for att in stg.nodes[node_from_same_cluster]:
                        if att.startswith("cluster-"):
                            stg.nodes[node][att] = stg.nodes[node_from_same_cluster][att]
            # Otherwise, we set it aside for merging with a new higher-level cluster.
            else:
                nodes_to_merge.extend(cluster_of_nodes)

        # If the next level already exists, we want to assign new nodes to clusters in it.
        if num_existing_levels > level + 1:
            # If there are no nodes to merge, return the current stg.
            if len(nodes_to_merge) == 0:
                return stg
            else:
                return self.assign_new_higher_level_clusters(stg, nodes_to_merge, level + 1, num_existing_levels)
        else:
            last_level_modularity = nx_comm.modularity(stg, self._get_clusters_from_level(stg, level - 1), weight=None)
            if best_modularity < last_level_modularity or math.isclose(
                last_level_modularity, best_modularity, abs_tol=1e-6
            ):
                if num_existing_levels == level + 1:
                    return stg
                else:
                    return original_stg
            else:
                if len(nodes_to_merge) == 0:
                    return stg
                elif num_existing_levels <= level + 1:
                    return self.assign_new_higher_level_clusters(stg, list(stg.nodes), level + 1, num_existing_levels)
                else:
                    return self.assign_new_higher_level_clusters(stg, nodes_to_merge, level + 1, num_existing_levels)

    def _reassign_disconnected_clusters(self, stg: nx.DiGraph):
        number_of_levels = self._get_current_number_of_levels(stg)

        # At each level of the hierarchy, we look at each cluster.
        for level in range(number_of_levels):
            existing_clusters = self._get_existing_clusters(stg, level)
            for cluster in existing_clusters:
                # Create the subgraph containing only nodes in this cluster.
                nodes_in_cluster = [node for node in stg.nodes if stg.nodes[node][f"cluster-{level}"] == cluster]
                sub_stg = stg.subgraph(nodes_in_cluster)

                # Get the number of weakly connected components in this subgraph. If it is
                # greater than one, give each connected component a unique cluster label.
                if nx.number_weakly_connected_components(sub_stg) > 1:
                    # Get list of all connected components, sorted in descending size order.
                    connected_components = [
                        component
                        for component in sorted(nx.weakly_connected_components(sub_stg), key=len, reverse=True)
                    ]

                    for i, component in enumerate(connected_components):
                        # Nodes in the first (i.e., largest) connected component gets to keep its current label.
                        if i == 0:
                            for node in component:
                                stg.nodes[node][f"cluster-{level}"] = cluster
                        # Nodes in other connected components get assigned a new label.
                        else:
                            new_id = max(self._get_existing_clusters(stg, level), default=-1) + 1
                            for node in component:
                                stg.nodes[node][f"cluster-{level}"] = new_id
        return stg

    def _get_clusters_from_level(self, stg: nx.DiGraph, level: int):
        return [
            list(map(operator.itemgetter(0), v))
            for k, v in itertools.groupby(
                nx.get_node_attributes(stg, f"cluster-{level}").items(),
                operator.itemgetter(1),
            )
        ]

    def _get_current_number_of_levels(self, stg: nx.DiGraph):
        level_attributes = set(itertools.chain(*[(stg.nodes[n].keys()) for n in stg.nodes()]))
        level_attributes = {level_att for level_att in level_attributes if level_att.startswith("cluster-")}
        return len(level_attributes)

    def _get_existing_clusters(self, stg, level):
        return set(nx.get_node_attributes(stg, f"cluster-{level}").values())


if __name__ == "__main__":
    num_runs = 40
    for run in range(num_runs):
        output_directory = "./Training Results/Rooms/Incremental Update/"
        output_directory_training = "./Training Results/Rooms/Incremental Update Training/"
        experiment_id = random.randrange(10000)
        num_agents = 1
        try:
            # Run Macro-Q Learning Agent
            for agent in range(num_agents):
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
                    n_step_updates=False,
                )

                train_results, test_results = agent.run_agent(
                    num_epochs=100,
                    epoch_length=100,
                    process_new_nodes_intervals=[100, 500, 1000, 3000, 5000],
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
