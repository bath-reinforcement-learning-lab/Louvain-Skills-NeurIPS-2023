import copy

from typing import Hashable
import networkx as nx

import random

from simpleoptions import BaseOption, BaseEnvironment


class SubgoalOption(BaseOption):
    def __init__(
        self,
        env: "BaseEnvironment",
        stg: nx.DiGraph,
        subgoal: Hashable,
        initiation_set: set,
        q_table: dict = None,
    ):
        """_summary_

        Args:
            stg (`DiGraph`): _description_
            subgoal (`Hashable`): _description_
            initiation_set (`set`): _description_
            q_table (`dict`, optional): _description_. Defaults to None, in which case an empty dictionary is used.
        """
        self.env = copy.copy(env)
        self.stg = stg
        self.subgoal = subgoal
        self.initiation_set = initiation_set

        if q_table is None:
            self.q_table = {}
        else:
            self.q_table = q_table

    def initiation(self, state):
        # Option can only be initiated from its source cluster.
        return state in self.initiation_set

    def policy(self, state, test=False):
        # Return highest-valued option from the Q-table, breaking ties randomly.
        available_options = self.env.get_available_options(state)
        max_value = max([self.q_table.get((hash(state), hash(option)), 0) for option in available_options])

        chosen_option = random.choice(
            [option for option in available_options if self.q_table[(hash(state), hash(option))] == max_value]
        )
        return chosen_option

    def termination(self, state):
        # Option terminates upon reaching the subgoal or leaving the initiation set.
        return float((state == self.subgoal) or (state not in self.initiation_set))

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
