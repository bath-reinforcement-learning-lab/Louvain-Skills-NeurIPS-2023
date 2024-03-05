import copy

from typing import Hashable
import networkx as nx

import random

from simpleoptions import BaseOption, BaseEnvironment


class LouvainOption(BaseOption):
    def __init__(
        self,
        env: "BaseEnvironment",
        stg: nx.DiGraph,
        hierarchy_level: int,
        source_cluster: Hashable,
        target_cluster: Hashable,
        can_leave_initiation_set: bool,
        q_table: dict = None,
    ):
        """
        Initiates a new LouvainOption, which takes an agent from some source clsuter to some target cluster.

        Args:
            stg (`DiGraph`): A state-transition graph representing the agent's environment.
            source_cluster (`Hashable`): The cluster in which the option can be initiated.
            target_cluster (`Hashable`): The cluster to which the option takes the agent.
            hierarchy_level (`int`): The level of the hierarchy this skill exists at.
            can_leave_initation_set (`bool`): Whether or not the agent can leave the source_cluster while navigating to the target_cluster.
            q_table (`dict`, optional): _description_. Defaults to None, in which case an empty dictionary is used.
        """
        self.env = copy.copy(env)
        self.stg = stg
        self.hierarchy_level = hierarchy_level
        self.source_cluster = source_cluster
        self.target_cluster = target_cluster
        self.can_leave_initiation_set = can_leave_initiation_set

        self.initiation_set = self._generate_initiation_set()
        if self.can_leave_initiation_set:
            self.executable_set = self._generate_executable_set()

        if q_table is None:
            self.q_table = {}
        else:
            self.q_table = q_table

    def initiation(self, state):
        # If this node has not been yet recorded on the STG, it cannot be in the initiation set.
        if state not in self.stg.nodes:
            return False
        # If this node has not yet been assigned a cluster at the current level of the hierarchy, it cannot be in the initiation set.
        elif f"cluster-{self.hierarchy_level}" not in self.stg.nodes[state]:
            return False
        # Else, test whether this state is in the initiation set.
        else:
            # return state in self.initiation_set
            # Pre-computing initiation sets saves a lot of time, but I'm reverting to the code below for now.
            # It's more expensive, but I need to find a nice way to handle both directed edges and the possibility
            # for transfer between tasks (a terminal state in one task may not be a terminal state for another,
            # causing issues when it comes to computing paths etc.).
            return self.stg.nodes[state][f"cluster-{self.hierarchy_level}"] == self.source_cluster

    def policy(self, state, test=False):
        # Return highest-valued option from the Q-table, breaking ties randomly.
        available_options = self.env.get_available_options(state)
        max_value = max([self.q_table.get((hash(state), hash(option)), 0) for option in available_options])

        chosen_option = random.choice(
            [option for option in available_options if self.q_table[(hash(state), hash(option))] == max_value]
        )
        return chosen_option

    def termination(self, state):
        if self.can_leave_initiation_set:
            return float(self.stg.nodes[state][f"cluster-{self.hierarchy_level}"] == self.target_cluster)
        else:
            return float(not self.initiation(state))

    def _generate_initiation_set(self):
        # Create sub-graph consisting of only the source cluster and the target cluster.
        source_nodes = [
            node
            for node in self.stg.nodes
            if self.stg.nodes[node][f"cluster-{self.hierarchy_level}"] == self.source_cluster
        ]
        target_nodes = [
            node
            for node in self.stg.nodes
            if self.stg.nodes[node][f"cluster-{self.hierarchy_level}"] == self.target_cluster
        ]
        subgraph = self.stg.subgraph(source_nodes + target_nodes)

        # Return the set of nodes in the starting clsuter from which there exists a path to the
        # target cluster *without leaving the source cluster*.
        return set(
            [
                node
                for node in source_nodes
                if any(nx.has_path(subgraph, node, target_node) for target_node in target_nodes)
            ]
        )

    def _generate_executable_set(self):
        # Generate set of nodes in the target cluster.
        target_nodes = [
            node
            for node in self.stg.nodes
            if self.stg.nodes[node][f"cluster-{self.hierarchy_level}"] == self.target_cluster
        ]

        # Return the set of nodes from which there exists a path to any node the target cluster.
        return set(
            [
                node
                for node in self.stg.nodes
                if node not in target_nodes
                and any(nx.has_path(self.stg, node, target_node) for target_node in target_nodes)
            ]
        )

    def __str__(self):
        return f"{self.hierarchy_level},{self.source_cluster},{self.target_cluster}"

    def __repr__(self):
        return f"LouvainOption(Level: {self.hierarchy_level}, Source: {self.source_cluster}, Target: {self.target_cluster})"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if isinstance(other, LouvainOption):
            return (
                self.hierarchy_level == other.hierarchy_level
                and self.source_cluster == other.source_cluster
                and self.target_cluster == other.target_cluster
            )
        else:
            return False

    def __ne__(self, other):
        return not self == other
