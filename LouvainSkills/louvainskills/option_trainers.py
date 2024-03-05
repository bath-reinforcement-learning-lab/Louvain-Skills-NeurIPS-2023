import math
import copy
import random
import numpy as np
import igraph as ig
import networkx as nx

from itertools import cycle
from collections import defaultdict

from simpleoptions import PrimitiveOption
from simpleenvs.envs.playroom import PlayroomEnvironment

from louvainskills.options import LouvainOption


class OptionTrainer(object):
    def __init__(self):
        pass

    def train_option_policy(self):
        pass


class LouvainOptionTrainer(OptionTrainer):
    def __init__(
        self,
        env,
        stg,
        epsilon=0.2,
        alpha=0.4,
        gamma=1.0,
        max_steps=100_000,
        max_episode_steps=250,
        can_leave_initiation_set=False,
        max_num_episodes=20_000,
    ):
        self.env = env
        self.stg = stg
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.max_steps = max_steps
        self.max_episode_steps = max_episode_steps
        self.can_leave_initiation_set = can_leave_initiation_set
        self.max_num_episodes = max_num_episodes

    def train_option_policy(
        self, hierarchy_level, source_cluster, target_cluster, q_table=None, default_action_value=0.0
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

        # Create a cycle over all states in the initiation set that we can sample initial states from when training.
        initiation_set_cycle = initiation_set * 3
        random.shuffle(initiation_set_cycle)
        initiation_set_cycle = cycle(initiation_set_cycle)

        time_steps = 0
        episodes = 0
        while time_steps < self.max_steps:
            # Choose (non-terminal!) initial state in the source cluster.
            state = self.env.reset(self._choose_initial_state(initiation_set_cycle))
            episode_steps = 0
            terminal = False

            while not terminal:
                # Select option based on current policy (derived using epsilon-greedy).
                option = self._select_option(state)

                # Run option until termination, tracking visited states and earned rewards.
                states = [state]
                rewards = []
                option_terminated = False
                while (not option_terminated) and (not terminal):
                    action = self._get_primitive_option(state, option).policy(state)
                    next_state, _, done, _ = self.env.step(action)
                    time_steps += 1
                    episode_steps += 1

                    # Compute reward and terminality.
                    if (
                        self.stg.nodes[next_state][f"cluster-{hierarchy_level}"] == target_cluster
                    ):  # Agent reaches the target cluster.
                        reward = 1.0
                        terminal = True
                    elif done:  # Agent reaches a terminal state.
                        reward = -1.0
                        terminal = True
                    elif (
                        (not self.can_leave_initiation_set)
                        and (self.stg.nodes[next_state][f"cluster-{hierarchy_level}"] != source_cluster)
                        and (self.stg.nodes[next_state][f"cluster-{hierarchy_level}"] != target_cluster)
                    ):  # Agent leaves the initiation set (and isn't allowed to do so!).
                        reward = -1.0
                        terminal = True
                    else:  # Otherwise...
                        reward = -0.001
                        terminal = False

                    option_terminated = bool(option.termination(next_state))

                    states.append(next_state)
                    rewards.append(reward)

                    state = next_state

                    # Training time-limit exceeded.
                    if (episode_steps > self.max_episode_steps) or (time_steps > self.max_steps):
                        break

                # Perform a Macro Q-Learning update for each state visited while the option was executing.
                self._macro_q_learn(states, rewards, option, terminal)

            # Check to see whether the episode limit has been reached.
            episodes += 1
            if episodes > self.max_num_episodes:
                break

        # For Debugging - adds policy labels to the STG.
        for state in initiation_set:
            if not ((state is None) or (self.env.is_state_terminal(state))):
                selected_option = self._select_option(state, test=True)
                self.stg.nodes[state][
                    f"{hierarchy_level},{source_cluster},{target_cluster}"
                ] = self._get_primitive_option(state, selected_option).action

        return self.q_table

    def _macro_q_learn(self, states, rewards, option, terminal, n_step=True):
        termination_state = states[-1]

        for i in range(len(states) - 1):
            initiation_state = states[i]

            old_value = self.q_table[(hash(initiation_state), hash(option))]

            # Compute discounted sum of rewards.
            discounted_sum_of_rewards = self._discounted_return(rewards[i:], self.gamma)

            # Get Q-Values for Next State.
            if not terminal:
                q_values = [
                    self.q_table[(hash(termination_state), hash(o))]
                    for o in self.env.get_available_options(termination_state)
                ]
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


class BetweennessOptionTrainer(OptionTrainer):
    def __init__(
        self,
        env,
        stg,
        epsilon=0.2,
        alpha=0.4,
        gamma=1.0,
        max_steps=1000,
        max_episode_steps=200,
        max_num_episodes=20_000,
    ):
        self.env = env
        self.stg = stg
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.max_steps = max_steps
        self.max_episode_steps = max_episode_steps
        self.max_num_episodes = max_num_episodes

    def train_option_policy(self, subgoal, initiation_set_size, q_table=None, default_action_value=0.0):
        # Initialise Q-table.
        if q_table is None:
            self.q_table = defaultdict(lambda: default_action_value)
        else:
            self.q_table = copy.deepcopy(q_table)

        # Define initiation set as the n closest nodes that have a path to the
        # subgoal node, where n = initiation_set_size.
        initiation_set = sorted(list(nx.single_target_shortest_path_length(self.stg, subgoal)), key=lambda x: x[1])
        initiation_set = list(list(zip(*initiation_set))[0])[1 : min(initiation_set_size + 1, len(initiation_set) - 1)]

        # Create a cycle over all states in the initiation set that we can sample initial states from when training.
        initiation_set_cycle = list(initiation_set) * 5
        random.shuffle(initiation_set_cycle)
        initiation_set_cycle = cycle(initiation_set_cycle)

        # Convert initiation set into a set for faster membership testing.
        initiation_set = set(initiation_set)

        time_steps = 0
        while time_steps < self.max_steps:
            # Choose (non-terminal!) state in the initiation set.
            state = next(
                self.env.reset(state)
                for state in initiation_set_cycle
                if not self.env.is_state_terminal(state) and state != subgoal
            )
            episode_steps = 0
            terminal = False

            while not terminal:
                # Select and execute action.
                action = self._select_action(state)
                next_state, _, done, _ = self.env.step(action.policy(state))
                time_steps += 1
                episode_steps += 1

                # Compute reward and terminality.
                if next_state == subgoal:  # Agent reached subgoal.
                    reward = 1.0
                    terminal = True
                elif next_state not in initiation_set:  # Agent left the initation set.
                    reward = -1.0
                    terminal = True
                elif done:  # Agent reached a terminal state.
                    reward = -1.0
                    terminal = True
                else:  # Otherwise...
                    reward = -0.001
                    terminal = False

                # Perform Q-Learning update.
                old_q = self.q_table[(hash(state), hash(action))]
                max_next_q = (
                    0
                    if terminal
                    else max(
                        [
                            self.q_table[(hash(next_state), hash(next_action))]
                            for next_action in self.env.get_available_options(next_state)
                        ]
                    )
                )
                new_q = reward + self.gamma * max_next_q
                self.q_table[(hash(state), hash(action))] = old_q + self.alpha * (new_q - old_q)

                state = next_state

                # Training time-limit exceeded.
                if (episode_steps > self.max_episode_steps) or (time_steps > self.max_steps):
                    break

            # Check to see whether the episode limit has been reached.
            episodes += 1
            if episodes > self.max_episode_limit:
                break

        # For Debugging - adds policy labels to the STG.
        for state in initiation_set:
            if not ((state is None) or (self.env.is_state_terminal(state))):
                selected_option = self._select_action(state, test=True)
                self.stg.nodes[state][f"{subgoal}"] = selected_option.policy(state)

        return self.q_table, initiation_set

    def _select_action(self, state, test=False):
        available_options = [option for option in self.env.get_available_options(state)]

        # Choose an exploratory action with probability epsilon.
        if (not test) and (random.random() < self.epsilon):
            return random.choice(available_options)
        # Choose the optimal action with probability (1 - epsilon), breaking ties randomly.
        else:
            max_value = max([self.q_table[(hash(state), hash(option))] for option in available_options])
            best_actions = [
                option for option in available_options if self.q_table[(hash(state), hash(option))] == max_value
            ]
            return random.choice(best_actions)
