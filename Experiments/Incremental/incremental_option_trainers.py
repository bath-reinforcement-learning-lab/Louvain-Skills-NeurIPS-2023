import copy
import math
import random

import numpy as np
import igraph as ig
import networkx as nx

from itertools import cycle
from collections import defaultdict

from simpleoptions import PrimitiveOption
from simpleoptions.environment import BaseEnvironment

from louvainskills.options import LouvainOption
from louvainskills.option_trainers import OptionTrainer
from louvainskills.utils.graph_layouts import gridlayout
from louvainskills.utils.graph_utils import get_all_neighbours


class IncrementalLouvainOptionTrainer(OptionTrainer):
    def __init__(
        self,
        env,
        stg,
        n_episodes=1000,
        epsilon=0.2,
        alpha=0.4,
        gamma=1.0,
        max_episode_length=200,
        can_leave_initiation_set=False,
    ):
        self.env = env
        self.stg = stg
        self.n_episodes = n_episodes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.max_episode_length = max_episode_length
        self.can_leave_initiation_set = can_leave_initiation_set

    def train_option_policy(
        self,
        hierarchy_level,
        source_cluster,
        target_cluster,
        q_table=None,
        default_action_value=0,
    ):
        # Initialise Q-table.
        if q_table is None:
            self.q_table = defaultdict(lambda: default_action_value)
        else:
            self.q_table = copy.deepcopy(q_table)

        # Define set of states in the source cluster, where this option can be invoked.
        initiation_set = [
            state for state, data in self.stg.nodes(data=True) if data[f"cluster-{hierarchy_level}"] == source_cluster
        ]

        # If the environment is Playroom, we can use a helper function to derive the skill hierarchy.

        # Create a cycle over all states in the initiation set that we can sample
        # initial states from when training.
        initiation_set_cycle = initiation_set * 3
        random.shuffle(initiation_set_cycle)
        initiation_set_cycle = cycle(initiation_set_cycle)

        # For N episodes.
        for episode in range(self.n_episodes):
            # Select random initial state in source cluster.
            state = self.env.reset(self._choose_initial_state(initiation_set_cycle))
            terminal = False

            time_steps = 0

            # Train until a state in the target cluster or a terminal state is reached.
            while (
                not terminal
                # and (not self.stg.nodes[state][f"cluster-{hierarchy_level}"] == target_cluster)
                # and (state in self.stg.nodes)
            ):
                # Select option based on current policy (derived using epsilon-greedy).
                option = self._select_option(state)

                # Run option until termination, tracking states visited and rewards earned.
                states = [copy.deepcopy(state)]
                rewards = []
                while not terminal:
                    # Execute action in current state, observe next-state and whether or not it is terminal.
                    action = self._get_primitive_option(state, option).policy(state)
                    next_state, __, terminal, __ = self.env.step(action)

                    time_steps += 1

                    # We reward our agent -0.001 per time-step until it reaches the target cluster.
                    # Reaches a terminal state.
                    if terminal:
                        reward = -1.0
                    # Attempts to leave the known STG.
                    elif next_state not in self.stg.nodes:
                        reward = -1.0
                        terminal = True
                    # Reaches the target cluster.
                    elif self.stg.nodes[next_state][f"cluster-{hierarchy_level}"] == target_cluster:
                        reward = 1.0
                        terminal = True
                    # Leaves the initiation set without reaching the target cluster.
                    elif (
                        (not self.can_leave_initiation_set)
                        and (self.stg.nodes[next_state][f"cluster-{hierarchy_level}"] != source_cluster)
                        and (self.stg.nodes[next_state][f"cluster-{hierarchy_level}"] != target_cluster)
                    ):
                        reward = -1.0
                        terminal = True
                    else:
                        reward = -0.001

                    # Save state and reward to list.
                    states.append(copy.deepcopy(next_state))
                    rewards.append(reward)

                    state = next_state

                    if time_steps > self.max_episode_length:
                        terminal = True

                    if (terminal) or (option.termination(state)):
                        break

                # Perform a Macro Q-Learning update for each state visited while the option was executing.
                self._macro_q_learn(states, rewards, option)

        # For Debugging - adds policy labels to the STG.
        for state in initiation_set:
            if not ((state is None) or (self.env.is_state_terminal(state))):
                selected_option = self._select_option(state, test=True)
                self.stg.nodes[state][f"{hierarchy_level},{source_cluster},{target_cluster}"] = (
                    self._get_primitive_option(state, selected_option).action
                )

        return self.q_table

    def _macro_q_learn(self, states, rewards, option, n_step=True):
        termination_state = states[-1]

        for i in range(len(states) - 1):
            initiation_state = states[i]

            old_value = self.q_table[(hash(initiation_state), hash(option))]

            # Compute discounted sum of rewards.
            discounted_sum_of_rewards = self._discounted_return(rewards[i:], self.gamma)

            # Get Q-Values for Next State.
            available_options = self.env.get_available_options(termination_state)

            if (not self.env.is_state_terminal(termination_state)) and (len(available_options) > 0):
                q_values = [self.q_table[(hash(termination_state), hash(o))] for o in available_options]
            # Cater for terminal states (Q-value is zero).
            else:
                q_values = [0]

            # Perform Macro-Q Update
            self.q_table[(hash(initiation_state), hash(option))] = old_value + self.alpha * (
                discounted_sum_of_rewards + math.pow(self.gamma, len(rewards) - i) * max(q_values) - old_value
            )

            # If we're not performing n-step updates, exit after the first iteration.
            if not n_step:
                break

    def _select_option(self, state, test=False):
        available_options = [option for option in self.env.get_available_options(state)]

        # Choose an exploratory action with probability epsilon.
        if (test == False) and (random.random() < self.epsilon):
            return random.choice(available_options)
        # Choose the optimal action with probability (1 - epsilon), breaking ties randomly.
        else:
            max_value = max([self.q_table[(hash(state), hash(option))] for option in available_options])
            best_actions = [
                option for option in available_options if self.q_table[(hash(state), hash(option))] == max_value
            ]
            return random.choice(best_actions)

    def _choose_initial_state(self, initiation_set_cycle):
        initial_state = None

        while (initial_state is None) or (self.env.is_state_terminal(initial_state)):
            initial_state = next(initiation_set_cycle)

        return initial_state

    def _get_primitive_option(self, state, option):
        # Recursively query the option policy until we get a primitive option.
        if isinstance(option, PrimitiveOption):
            return option
        else:
            return self._get_primitive_option(state, option.policy(state))

    def _discounted_return(self, rewards, gamma):
        # Computes the discounted reward given an ordered list of rewards, and a discount factor.
        num_rewards = len(rewards)

        # Fill an array with gamma^index for index = 0 to index = num_rewards - 1.
        gamma_exp = np.power(np.full(num_rewards, gamma), np.arange(0, num_rewards))

        # Element-wise multiply and then sum array.
        discounted_sum_of_rewards = np.sum(np.multiply(rewards, gamma_exp))

        return discounted_sum_of_rewards
