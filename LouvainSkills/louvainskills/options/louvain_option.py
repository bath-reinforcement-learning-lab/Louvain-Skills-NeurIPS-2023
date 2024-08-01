import networkx as nx

from typing import Hashable

from simpleoptions import PseudoRewardOption


class LouvainOption(PseudoRewardOption):
    def __init__(
        self,
        stg: nx.DiGraph,
        hierarchy_level: int,
        source_cluster: Hashable,
        target_cluster: Hashable,
        can_leave_initiation_set: bool,
        policy_dict: dict = None,
    ):
        """
        Initiates a new LouvainOption, which takes an agent from some source clsuter to some target cluster.

        Args:
            stg (`DiGraph`): A state-transition graph representing the agent's environment.
            source_cluster (`Hashable`): The cluster in which the option can be initiated.
            target_cluster (`Hashable`): The cluster to which the option takes the agent.
            hierarchy_level (`int`): The level of the hierarchy this skill exists at.
            can_leave_initation_set (`bool`): Whether or not the agent can leave the source_cluster while navigating to the target_cluster.
            policy (`dict`, optional): The dictionary used to represent the option's policy. A mapping from state to Option. Defaults to None, in which case an empty dictionary is used.
        """
        self.stg = stg
        self.hierarchy_level = hierarchy_level
        self.source_cluster = source_cluster
        self.target_cluster = target_cluster
        self.can_leave_initiation_set = can_leave_initiation_set

        self.initiation_set = self._generate_initiation_set()
        if self.can_leave_initiation_set:
            self.executable_set = self._generate_executable_set()
        else:
            self.executable_set = self.initiation_set

        if policy_dict is None:
            self.policy_dict = {}
        else:
            self.policy_dict = policy_dict

    def initiation(self, state):
        # If this node has not been yet recorded on the STG, it cannot be in the initiation set.
        if state not in self.stg.nodes:
            return False
        # If this node has not yet been assigned a cluster at the current level of the hierarchy, it cannot be in the initiation set.
        elif f"cluster-{self.hierarchy_level}" not in self.stg.nodes[state]:
            return False
        # Else, test whether this state is in the initiation set.
        else:
            return state in self.initiation_set

    def policy(self, state, test=False):
        return self.policy_dict[state]

    def termination(self, state):
        if not self.can_leave_initiation_set:
            return float(state not in self.initiation_set)
        else:
            return float(state not in self.initiation_set and state not in self.executable_set)

    def pseudo_reward(self, state: Hashable, action: Hashable, next_state: Hashable) -> float:
        # If the agent is allowed to leave the initation set, it is allowed to reach the target cluster via any node.
        if self.can_leave_initiation_set:
            # If the agent is in the set of states from which it can reach the target cluster, it earns a small negative reward of -0.001.
            if next_state in self.executable_set:
                return -0.001
            else:
                # If the agent has reached the target cluster, it earns a reward of 1.0.
                if self.stg.nodes[next_state][f"cluster-{self.hierarchy_level}"] == self.target_cluster:
                    return 1.0
                # If the agent has left the set of states from which it can reach the target cluster, it earns a large negative reward of -1.0.
                else:
                    return -1.0
        # If the agent is not allowed to leave the initiation set, it has to reach the target cluster via a path that remains in the source cluster.
        else:
            # If the agent is in the initiation set, it earns a small negative reward of -0.001.
            if next_state in self.initiation_set:
                return -0.001
            else:
                # If the agent has reached the target cluster, it earns a reward of 1.0.
                if self.stg.nodes[next_state][f"cluster-{self.hierarchy_level}"] == self.target_cluster:
                    return 1.0
                # Otherwise, it has left the source cluster and entered a non-target cluster, earning a large negative reward of -1.0.
                else:
                    return -1.0

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

        # Return the set of nodes in the starting cluster from which there exists a path to the
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
