import math
import copy
import random

import numpy as np
import networkx as nx

from simpleoptions import BaseEnvironment, BaseOption, PrimitiveOption

from collections import defaultdict
from typing import Dict


class EigenOption(BaseOption):
    def __init__(
        self,
        env: BaseEnvironment,
        stg: nx.Graph,
        pvf: Dict,
        eigenoption_index: int,
    ):
        """
        ASSUMES A DETERMINISTIC ENVIRONMENT.

        Args:
            env (`BaseEnvironment`)
            pvf (`dict`): _description_
            eigenoption_index (`int`): _description_
        """
        self.stg = stg
        self.env = copy.copy(env)
        self.pvf = pvf
        self.eigenoption_index = eigenoption_index

        self.primitive_actions = {
            option.action: option for option in env.get_option_space() if isinstance(option, PrimitiveOption)
        }

    def initiation(self, state):
        # Option is available everywhere it doesn't terminate.
        return not self.termination(state)

    def policy(self, state, test=False):
        # Return the action which maximises the expected return in this state, with
        # respect to the intrinstic reward function defined by the proto-value function.
        # print(list(self.primitive_actions.keys()))
        if self.primitive_policy[state] == "EIG_TERMINATE":
            print(self.initiation(state))
            print(self.termination(state))
            print(state)
        return self.primitive_actions[self.primitive_policy[state]]

    def termination(self, state):
        # Eigenoptions terminate when the terminate action is executed.
        if self.env.is_state_terminal(state) or self.primitive_policy[state] == "EIG_TERMINATE":
            return float(True)
        else:
            return float(False)

    def train(self, gamma=0.9, theta=1e-8):
        policy = {}
        values = {}

        # Initialise values and policy.
        for state in self.pvf.keys():
            values[state] = 0

            if not self.env.is_state_terminal(state):
                policy[state] = "EIG_TERMINATE"

        while True:
            # Policy Evaluation Step.
            while True:
                delta = 0
                for state in self.pvf.keys():
                    if self.env.is_state_terminal(state):
                        continue

                    v_old = values[state]

                    action = policy[state]
                    if action == "EIG_TERMINATE":
                        next_state = "EIG_TERMINAL"
                        reward = 0
                    else:
                        (next_state, _), _ = self.env.get_successors(state, [action])[0]
                        reward = self._intrinsic_reward(state, next_state)

                    if next_state == "EIG_TERMINAL" or self.env.is_state_terminal(next_state):
                        v_next = 0
                    else:
                        v_next = values[next_state]

                    values[state] = reward + gamma * v_next

                    delta = max(delta, abs(v_old - values[state]))
                if delta < theta:
                    break

            # Policy Improvement Step.
            policy_stable = True
            for state in self.pvf.keys():
                if self.env.is_state_terminal(state):
                    continue

                a_old = policy[state]

                best_action = None
                best_value = -np.inf
                for action in self._get_available_primitives(state):
                    if action == "EIG_TERMINATE":
                        next_state = "EIG_TERMINAL"
                        reward = 0
                    else:
                        (next_state, _), _ = self.env.get_successors(state, [action])[0]
                        reward = self._intrinsic_reward(state, next_state)

                    if next_state == "EIG_TERMINAL":
                        v_next = 0
                    elif self.env.is_state_terminal(next_state):
                        continue
                    else:
                        v_next = values[next_state]

                    value = reward + gamma * v_next

                    if value > best_value:
                        best_action = action
                        best_value = value

                policy[state] = best_action

                if a_old != policy[state]:
                    policy_stable = False

            if policy_stable:
                break

        # Add policy to graph for inspection.
        for state in self.stg.nodes:
            if state in values:
                self.stg.nodes[state][f"PVF {self.eigenoption_index} Values"] = values[state]
            if state in policy:
                self.stg.nodes[state][f"PVF {self.eigenoption_index} Policy"] = str(policy[state])

        self.values = values
        self.primitive_policy = policy

    def _get_available_primitives(self, state):
        return ["EIG_TERMINATE"] + [action for action in self.env.get_available_actions(state)]

    def _intrinsic_reward(self, state, next_state):
        reward = self.pvf[next_state] - self.pvf[state]

        if math.isclose(reward, 0, abs_tol=1e-6):
            return 0
        else:
            return reward

    def __str__(self):
        return f"EigenOption({self.eigenoption_index})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if isinstance(other, EigenOption):
            return self.eigenoption_index == other.eigenoption_index
        else:
            return False

    def __ne__(self, other):
        return not self == other
