import copy

from typing import Hashable
import networkx as nx

import random

from simpleoptions import PseudoRewardOption, BaseEnvironment


class SubgoalOption(PseudoRewardOption):
    def __init__(
        self,
        stg: nx.DiGraph,
        subgoal: Hashable,
        initiation_set: set,
        policy_dict: dict = None,
    ):
        """_summary_

        Args:
            stg (`DiGraph`): The state transition graph of the environment.
            subgoal (`Hashable`): The subgoal state of the option.
            initiation_set (`set`): The set of states from which the option can be initiated.
            policy_dict (`dict`, optional): A dictionary of policies for the option. Defaults to None, in which case an empty dictionary is used.
        """
        self.stg = stg
        self.subgoal = subgoal
        self.initiation_set = initiation_set

        if policy_dict is None:
            self.policy_dict = {}
        else:
            self.policy_dict = policy_dict

    def initiation(self, state):
        # Option can only be initiated from its source cluster.
        return state in self.initiation_set

    def policy(self, state, test=False):
        # Return highest-valued option from the Q-table, breaking ties randomly.
        return self.policy_dict[state]

    def termination(self, state):
        # Option terminates upon reaching the subgoal or leaving the initiation set.
        return float((state == self.subgoal) or (state not in self.initiation_set))

    def pseudo_reward(self, state: Hashable, action: Hashable, next_state: Hashable) -> float:
        # If the next state is the subgoal, the agent earns a reward of 1.0.
        if next_state == self.subgoal:
            return 1.0
        # If the agent has left the initiation set, it earns a large negative reward of -1.0.
        elif next_state not in self.initiation_set:
            return -1.0
        # Otherwise, it is still in the initiation set and earns a small negative reward of -0.001.
        else:
            return -0.001

    def __str__(self):
        return f"{self.subgoal},{len(self.initiation_set)}"

    def __repr__(self):
        return f"SubgoalOption(Subgoal: {self.subgoal}, Initiation Set Size: {len(self.initiation_set)})"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if isinstance(other, SubgoalOption):
            return self.subgoal == other.subgoal and len(self.initiation_set) == len(other.initiation_set)
        else:
            return False

    def __ne__(self, other):
        return not self == other
