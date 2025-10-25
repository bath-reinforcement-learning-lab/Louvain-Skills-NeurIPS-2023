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

from louvainskills.incremental_louvain import apply_incremental_louvain
from louvainskills.options import LouvainOption
from louvainskills.incremental_option_trainers import IncrementalValueIterationOptionTrainer

# Use the same testbed as the Replace agent for initial experiments.
from simpleenvs.envs.discrete_rooms import XuFourRooms


class IncrementalUpdateAgent(OptionAgent):
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
        Creates a new IncrementalUpdateAgent.

        This agent builds its state-transition graph (STG) online. At specified decision
        stages, it updates cluster assignments for only the newly observed nodes using
        an incremental Louvain procedure, pushes those labels up the hierarchy, and then
        (re)trains Louvain options over the updated partitions.

        Most existing skills carry forward naturally because partitions are updated
        rather than recomputed from scratch.
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
    ):
        """
        Trains the agent for a given number of epochs, with episodic evaluation support.
        """
        # Total number of primitive decision stages.
        num_time_steps = num_epochs * epoch_length

        # Reward logs at primitive decision-stage granularity.
        training_rewards = [None for _ in range(num_time_steps)]

        # Episodic evaluation scheduling.
        if test_interval > 0:
            test_interval_time_steps = test_interval * epoch_length
            episodic_evaluation_rewards = [None for _ in range(num_time_steps // test_interval_time_steps)]
            if self.test_env is None:
                raise RuntimeError("No test_env has been provided specified.")
        else:
            episodic_evaluation_rewards = []

        # Start with only primitive options.
        primitive_options = [PrimitiveOption(a, self.env) for a in self.env.get_action_space()]
        self.env.set_options(primitive_options)
        if self.test_env is not None:
            self.test_env.set_options(primitive_options)

        # Online STG construction.
        stg = nx.DiGraph()
        new_nodes: List = []

        episode = 0
        time_steps = 0

        while time_steps < num_time_steps:
            state = self.env.reset()
            terminal = False

            # Ensure the initial state is recorded.
            if not stg.has_node(state):
                stg.add_node(state)
                new_nodes.append(state)

            if render_interval > 0:
                self.env.render()

            # ──────────────────────────────────────────────────────────────────────
            # Run one episode.
            # ──────────────────────────────────────────────────────────────────────
            while not terminal:
                selected_option = self.select_action(state, self.executing_options)

                # Higher-level option selected: push on the execution stack.
                if isinstance(selected_option, BaseOption):
                    self.executing_options.append(copy.copy(selected_option))
                    self.executing_options_states.append([state])
                    self.executing_options_rewards.append([])

                # Primitive action selected: step environment and learn.
                else:
                    time_steps += 1
                    next_state, reward, terminal, __ = self.env.step(selected_option)

                    # Log rewards and (optionally) richer traces.
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

                    # Render if required.
                    if render_interval > 0 and time_steps % render_interval == 0:
                        self.env.render()

                    # Grow the discovered STG with the observed transition.
                    if not stg.has_node(next_state):
                        stg.add_node(next_state)
                        new_nodes.append(next_state)
                    if not stg.has_edge(state, next_state):
                        stg.add_edge(state, next_state)

                    state = next_state

                    # ──────────────────────────────────────────────────────────────
                    # Incremental partition + skill updates at specified time-steps.
                    # ──────────────────────────────────────────────────────────────
                    if time_steps in process_new_nodes_intervals:
                        print(f"Decision Stage {time_steps}/{num_time_steps}.")
                        if len(new_nodes) > 0:
                            print("Updating partitions incrementally...")
                            stg = apply_incremental_louvain(stg, new_nodes)
                            new_nodes = []
                            print("Updated partitions.")

                            print("Updating/Training option hierarchy...")
                            stg, options = self.update_options(stg)
                            self.env.set_options(options)
                            if self.test_env is not None:
                                self.test_env.set_options(options)
                            self.purge_old_q_table()
                            print("Updated option hierarchy.\n")

                    # Append state/reward to executing options’ trajectories.
                    for i in range(len(self.executing_options)):
                        self.executing_options_states[i].append(next_state)
                        self.executing_options_rewards[i].append(reward)

                    # Pop/learn any options that terminate on this time-step.
                    while self.executing_options and self._roll_termination(self.executing_options[-1], next_state):
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

                    # Episodic evaluation at interval boundaries.
                    if test_interval > 0 and time_steps % test_interval_time_steps == 0:
                        episodic_evaluation_rewards[(time_steps - 1) // test_interval_time_steps] = self.test_policy(
                            test_length=test_length,
                            test_runs=test_runs,
                            eval_number=time_steps // test_interval_time_steps,
                            allow_exploration=False,
                            verbose_logging=verbose_logging,
                            episodic_eval=True,
                        )

                # Cut the episode if we have reached the total training budget.
                if time_steps >= num_time_steps:
                    terminal = True

                # Flush any options still executing at episode end.
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

        # Return logs like the Replace agent.
        if verbose_logging:
            return self.training_log, (self.episodic_evaluation_log if self.episodic_evaluation_log else None)
        else:
            training_epoch_rewards = [
                sum(training_rewards[i * epoch_length : (i + 1) * epoch_length]) for i in range(num_epochs)
            ]
            return training_epoch_rewards, (episodic_evaluation_rewards if episodic_evaluation_rewards else None)

    # ──────────────────────────────────────────────────────────────────────────────
    # Skills over current partitions (directed).
    # ──────────────────────────────────────────────────────────────────────────────
    def update_options(self, stg: nx.DiGraph):
        """
        Builds (or refreshes) Louvain options over the **current** partition labels:
        for each level i and each directed edge (u -> v) at the primitive level, if
        cluster_i(u) != cluster_i(v), we create a directed skill (i, cluster_i(u), cluster_i(v)).

        Options are trained using IncrementalValueIterationOptionTrainer (deterministic case
        uses a single rollout).
        """
        # Primitive options for this environment.
        primitive_options = [PrimitiveOption(a, self.env) for a in self.env.get_action_space()]

        # Determine the number of levels from present cluster-* labels.
        num_levels = self._get_current_number_of_levels(stg)

        # Build a directed skill set per level from observed transitions.
        skill_hierarchy: List[set] = []
        for i in range(num_levels):
            directed_skills = set()
            # Use directed edges observed in the STG (u -> v).
            for u, v in stg.edges():
                cu = stg.nodes[u].get(f"cluster-{i}")
                cv = stg.nodes[v].get(f"cluster-{i}")
                if cu is None or cv is None:
                    continue
                if cu != cv:
                    directed_skills.add((i, cu, cv))
            skill_hierarchy.append(directed_skills)

        # Fresh copy of the environment for training options.
        training_env = self.env.__class__()
        training_env.reset()

        options: List[List[LouvainOption]] = []
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

            # Available options at this level = previously trained level or primitives at level 0.
            if level == 0:
                training_env.set_options(copy.copy(primitive_options))
            else:
                training_env.set_options(copy.copy(options[level - 1]))

            # Train this level’s skills.
            for i, src_cluster, dst_cluster in tqdm(hierarchy_level, desc="Training Skills"):
                opt = LouvainOption(
                    stg=stg,
                    hierarchy_level=i,
                    source_cluster=src_cluster,
                    target_cluster=dst_cluster,
                    can_leave_initiation_set=False,
                )

                # Skip if there’s no valid domain yet (e.g., empty initiation set).
                if not opt.initiation_set:
                    continue

                policy = option_trainer.train_option_policy(opt, can_leave_initiation_set=False)
                if not policy:
                    continue

                opt.policy_dict = policy
                options[level].append(opt)

        # Flatten + append primitives.
        flat_options = [o for lvl in options for o in lvl]
        flat_options.extend(primitive_options)

        return stg, flat_options

    def purge_old_q_table(self):
        """
        Removes Q-table entries associated with obsolete options after the skill set is refreshed.
        """
        primitive_hashes = [hash(PrimitiveOption(a, self.env)) for a in self.env.get_action_space()]
        current_option_hashes = set(hash(o) for o in self.env.get_option_space())

        keys_to_del = []
        for key in list(self.q_table.keys()):
            _, action_hash = key
            # Keep primitives and any option that still exists; drop anything else.
            if (action_hash not in primitive_hashes) and (action_hash not in current_option_hashes):
                keys_to_del.append(key)

        for key in keys_to_del:
            del self.q_table[key]

    @staticmethod
    def _get_current_number_of_levels(stg: nx.DiGraph) -> int:
        # Count unique "cluster-*" attributes present on any node.
        level_attrs = set()
        for n in stg.nodes():
            for att in stg.nodes[n].keys():
                if isinstance(att, str) and att.startswith("cluster-"):
                    level_attrs.add(att)
        return len(level_attrs)


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    # Target number of successful runs.
    target_successful_runs = 10

    successful_runs = 0
    attempts = 0

    env_name = "XuFourRooms"
    output_directory = "./Training Results/Chapter 3/Incremental/Rooms/Episode/Update/"
    output_directory_training = "./Training Results/Chapter 3/Incremental/Rooms/Train/Update/"

    while successful_runs < target_successful_runs:
        attempts += 1
        print(f"\n[Update] Attempt {attempts} (successful so far: {successful_runs}/{target_successful_runs})...")

        try:
            experiment_id = random.randrange(10000)

            # Deterministic environment.
            env = XuFourRooms(movement_penalty=-0.01, goal_reward=1.0)
            test_env = XuFourRooms(movement_penalty=-0.01, goal_reward=1.0)

            agent = IncrementalUpdateAgent(
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
            print(f"[Update] Run {successful_runs} of {target_successful_runs} completed successfully.")

        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break

        except Exception as e:
            # Log the failure and continue to the next attempt.
            print(f"[Update] Attempt {attempts} failed with an exception. Skipping this run.")
            print(f"Reason: {e}")
            traceback.print_exc()
            gc.collect()
            continue

    print(f"\n[Update] Finished. Successful runs: {successful_runs}/{target_successful_runs}.")
