# hybrid_agent.py (parity with Replace/Update)

import gc
import copy
import uuid
import json
import random
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import networkx.algorithms.community as nx_comm
from tqdm import tqdm

from simpleoptions import PrimitiveOption, OptionAgent
from simpleoptions.environment import BaseEnvironment
from simpleoptions.option import BaseOption

from louvainskills.louvain import apply_louvain
from louvainskills.incremental_louvain import apply_incremental_louvain
from louvainskills.options import LouvainOption
from louvainskills.utils.graph_utils import convert_nx_to_ig, convert_ig_to_nx
from louvainskills.incremental_option_trainers import IncrementalValueIterationOptionTrainer

from simpleenvs.envs.discrete_rooms import XuFourRooms


def compute_top_level_modularity(stg: nx.DiGraph) -> float:
    """
    Compute modularity for the highest 'cluster-*' level found on the STG.
    Tries to compute on the directed STG (matching your current pipeline).
    Falls back to an undirected view only if NetworkX complains.
    """
    if stg.number_of_nodes() == 0:
        return 0.0

    level_keys = {k for _, d in stg.nodes(data=True) for k in d if isinstance(k, str) and k.startswith("cluster-")}
    if not level_keys:
        return 0.0

    top_key = max(level_keys, key=lambda s: int(s.split("-")[1]))

    by_cluster = {}
    for n, d in stg.nodes(data=True):
        if top_key in d:
            by_cluster.setdefault(d[top_key], []).append(n)

    communities = [nodes for nodes in by_cluster.values() if nodes]
    if len(communities) <= 1:
        return 0.0

    try:
        return nx_comm.modularity(stg, communities, weight=None)
    except Exception:
        return nx_comm.modularity(stg.to_undirected(), communities, weight=None)


class IncrementalHybridAgent(OptionAgent):
    """
    Hybrid = Incremental Update most of the time, but if top-level modularity
    drops more than a threshold vs. the best since last Replace, rebuild from scratch.
    """

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
        replace_drop_threshold: float = 0.05,
    ):
        super().__init__(
            env, test_env, epsilon, macro_alpha, intra_option_alpha, gamma, default_action_value, n_step_updates
        )
        self.vi_theta = vi_theta
        self.vi_gamma = vi_gamma
        self.vi_num_rollouts = vi_num_rollouts
        self.vi_deterministic = vi_deterministic

        self.replace_drop_threshold = replace_drop_threshold

        self._first_update = True
        self._best_top_mod_since_replace: float = 0.0

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
        num_time_steps = num_epochs * epoch_length
        training_rewards = [None for _ in range(num_time_steps)]

        if test_interval > 0:
            test_interval_time_steps = test_interval * epoch_length
            episodic_evaluation_rewards = [None for _ in range(num_time_steps // test_interval_time_steps)]
            if self.test_env is None:
                raise RuntimeError("No test_env has been provided specified.")
        else:
            episodic_evaluation_rewards = []

        # primitives only to start
        options = [PrimitiveOption(a, self.env) for a in self.env.get_action_space()]
        self.env.set_options(options)
        if self.test_env is not None:
            self.test_env.set_options(options)

        stg = nx.DiGraph()
        new_nodes: List = []

        episode = 0
        time_steps = 0

        while time_steps < num_time_steps:
            state = self.env.reset()
            terminal = False

            if not stg.has_node(state):
                stg.add_node(state)
                new_nodes.append(state)

            if render_interval > 0:
                self.env.render()

            while not terminal:
                selected_option = self.select_action(state, self.executing_options)

                if isinstance(selected_option, BaseOption):
                    self.executing_options.append(copy.copy(selected_option))
                    self.executing_options_states.append([state])
                    self.executing_options_rewards.append([])
                else:
                    time_steps += 1
                    next_state, reward, terminal, __ = self.env.step(selected_option)

                    training_rewards[time_steps - 1] = reward
                    if verbose_logging:
                        transition = {
                            "state": state,
                            "next_state": next_state,
                            "reward": reward,
                            "terminal": terminal,
                            "active_options": [str(o) for o in self.executing_options],
                        }
                        for k, v in transition.items():
                            self.training_log[k].append(v)

                    if render_interval > 0 and time_steps % render_interval == 0:
                        self.env.render()

                    if not stg.has_node(next_state):
                        stg.add_node(next_state)
                        new_nodes.append(next_state)
                    if not stg.has_edge(state, next_state):
                        stg.add_edge(state, next_state)

                    state = next_state

                    # scheduled updates
                    if time_steps in process_new_nodes_intervals:
                        print(f"Decision Stage {time_steps}/{num_time_steps}.")
                        if len(new_nodes) > 0:
                            if self._first_update:
                                # first update: full replace
                                stg, options = self._rebuild_full_hierarchy(stg)  # resolution=1.0 (parity)
                                self.env.set_options(options)
                                if self.test_env is not None:
                                    self.test_env.set_options(options)
                                self.purge_all_non_primitives()
                                self._first_update = False
                                self._best_top_mod_since_replace = compute_top_level_modularity(stg)
                                print(
                                    f"[Hybrid] Full replace (initial). Top-level modularity = "
                                    f"{self._best_top_mod_since_replace:.6f}"
                                )
                                new_nodes = []
                            else:
                                # incremental update first
                                print("Updating partitions incrementally...")
                                stg = apply_incremental_louvain(stg, new_nodes)
                                new_nodes = []
                                current_mod = compute_top_level_modularity(stg)
                                self._best_top_mod_since_replace = max(self._best_top_mod_since_replace, current_mod)
                                print(
                                    f"[Hybrid] Incremental update: top-level modularity = {current_mod:.6f} "
                                    f"(best since replace = {self._best_top_mod_since_replace:.6f})"
                                )

                                # gate: if drop beyond threshold, full replace
                                if current_mod < (self._best_top_mod_since_replace - self.replace_drop_threshold):
                                    print("[Hybrid] Modularity drop beyond threshold → full replace.")
                                    stg, options = self._rebuild_full_hierarchy(stg)  # resolution=1.0 (parity)
                                    self.env.set_options(options)
                                    if self.test_env is not None:
                                        self.test_env.set_options(options)
                                    self.purge_all_non_primitives()
                                    self._best_top_mod_since_replace = compute_top_level_modularity(stg)
                                    print(
                                        f"[Hybrid] Full replace done. Top-level modularity = "
                                        f"{self._best_top_mod_since_replace:.6f}"
                                    )
                                else:
                                    # otherwise, just retrain options on current partitions
                                    stg, options = self._train_options_on_current_partitions(stg)
                                    self.env.set_options(options)
                                    if self.test_env is not None:
                                        self.test_env.set_options(options)
                                    self.purge_obsolete_q_table()
                                    print("[Hybrid] Retrained options on updated partitions.")

                    # option termination + learning
                    for i in range(len(self.executing_options)):
                        self.executing_options_states[i].append(next_state)
                        self.executing_options_rewards[i].append(reward)

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

                    # episodic evaluation
                    if test_interval > 0 and time_steps % test_interval_time_steps == 0:
                        episodic_evaluation_rewards[(time_steps - 1) // test_interval_time_steps] = self.test_policy(
                            test_length=test_length,
                            test_runs=test_runs,
                            eval_number=time_steps // test_interval_time_steps,
                            allow_exploration=False,
                            verbose_logging=verbose_logging,
                            episodic_eval=True,
                        )

                if time_steps >= num_time_steps:
                    terminal = True

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

        if verbose_logging:
            training_log = self.training_log
            evaluation_log = self.episodic_evaluation_log if self.episodic_evaluation_log else None
            return training_log, evaluation_log
        else:
            training_log = [sum(training_rewards[i * epoch_length : (i + 1) * epoch_length]) for i in range(num_epochs)]
            return training_log, (episodic_evaluation_rewards if episodic_evaluation_rewards else None)

    # ───────── helpers ─────────

    def _rebuild_full_hierarchy(self, original_stg: nx.DiGraph) -> Tuple[nx.DiGraph, List[LouvainOption]]:
        """
        Full Replace, parity with your Replace agent: resolution=1.0
        """
        stg = nx.DiGraph()
        stg.add_nodes_from(original_stg.nodes)
        stg.add_edges_from(original_stg.edges)

        stg_ig = convert_nx_to_ig(stg)
        stg_ig, aggs_ig = apply_louvain(
            stg_ig,
            resolution=1.0,
            return_aggregate_graphs=True,
            first_levels_to_skip=1,  # PARITY LINE
        )
        stg = convert_ig_to_nx(stg_ig)

        aggs = []
        for agg_ig in aggs_ig:
            agg = convert_ig_to_nx(agg_ig)
            if agg.number_of_nodes() > 1:
                aggs.append(copy.deepcopy(agg))

        skill_hierarchy = []
        for i, agg in enumerate(aggs[1:]):
            skill_hierarchy.append([])
            for u, v in agg.edges():
                if u != v:
                    skill_hierarchy[i].append((i, u, v))

        return stg, self._train_louvain_options(stg, skill_hierarchy)

    def _train_options_on_current_partitions(self, stg: nx.DiGraph) -> Tuple[nx.DiGraph, List[LouvainOption]]:
        """
        Retrain options over current labels — parity with IncrementalUpdateAgent.update_options:
        for each level i present on the STG and each directed primitive edge (u -> v),
        if cluster_i(u) != cluster_i(v), create a directed skill (i, cluster_i(u), cluster_i(v)).
        """
        # Primitive options for this environment.
        primitive_options = [PrimitiveOption(a, self.env) for a in self.env.get_action_space()]

        # Determine the number of levels from present cluster-* labels (same as Update).
        level_attrs = set()
        for n in stg.nodes():
            for att in stg.nodes[n].keys():
                if isinstance(att, str) and att.startswith("cluster-"):
                    level_attrs.add(att)
        # No cluster labels yet → keep current options (don’t regress to primitives).
        if not level_attrs:
            return stg, list(self.env.get_option_space())

        # Build a directed skill set per level from observed transitions (identical to Update).
        max_level = max(int(k.split("-")[1]) for k in level_attrs)
        skill_hierarchy: List[set] = []
        for i in range(max_level + 1):
            directed_skills = set()
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

            if level == 0:
                training_env.set_options(copy.copy(primitive_options))
            else:
                training_env.set_options(copy.copy(options[level - 1]))

            for i, src_cluster, dst_cluster in tqdm(hierarchy_level, desc="Training Skills"):
                opt = LouvainOption(
                    stg=stg,
                    hierarchy_level=i,
                    source_cluster=src_cluster,
                    target_cluster=dst_cluster,
                    can_leave_initiation_set=False,
                )
                if not opt.initiation_set:
                    continue

                policy = option_trainer.train_option_policy(opt, can_leave_initiation_set=False)
                if not policy:
                    continue

                opt.policy_dict = policy
                options[level].append(opt)

        flat_options = [o for lvl in options for o in lvl]
        flat_options.extend(primitive_options)
        return stg, flat_options

    def _train_louvain_options(
        self, stg: nx.DiGraph, skill_hierarchy: List[List[Tuple[int, int, int]]]
    ) -> List[LouvainOption]:
        primitive_options = [PrimitiveOption(a, self.env) for a in self.env.get_action_space()]

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

            if level == 0:
                training_env.set_options(copy.copy(primitive_options))
            else:
                training_env.set_options(copy.copy(options[level - 1]))

            for i, u, v in tqdm(hierarchy_level, desc="Training Skills"):
                option = LouvainOption(
                    stg=stg,
                    hierarchy_level=i,
                    source_cluster=u,
                    target_cluster=v,
                    can_leave_initiation_set=False,
                )
                if not option.initiation_set:
                    continue

                policy = option_trainer.train_option_policy(option, can_leave_initiation_set=False)
                if not policy:
                    continue

                option.policy_dict = policy
                options[level].append(option)

        flat = [o for lvl in options for o in lvl]
        flat.extend(primitive_options)
        return flat

    def purge_all_non_primitives(self):
        """
        Use only after a full Replace. Keeps primitives, drops all learned options.
        """
        primitive_hashes = [hash(PrimitiveOption(a, self.env)) for a in self.env.get_action_space()]
        keys_to_del = []
        for state_hash, action_hash in list(self.q_table.keys()):
            if action_hash not in primitive_hashes:
                keys_to_del.append((state_hash, action_hash))
        for k in keys_to_del:
            del self.q_table[k]

    def purge_obsolete_q_table(self):
        """
        Use after incremental updates. Keep primitives and any option that still exists.
        Delete only entries tied to options that are no longer present.
        """
        primitive_hashes = [hash(PrimitiveOption(a, self.env)) for a in self.env.get_action_space()]
        current_option_hashes = set(hash(o) for o in self.env.get_option_space())

        keys_to_del = []
        for state_hash, action_hash in list(self.q_table.keys()):
            # Drop if it's neither a primitive nor a still-present option.
            if (action_hash not in primitive_hashes) and (action_hash not in current_option_hashes):
                keys_to_del.append((state_hash, action_hash))
        for k in keys_to_del:
            del self.q_table[k]


# Runner (keep-until-success style, like Replace/Update)
if __name__ == "__main__":
    import traceback

    target_successful_runs = 10
    successful_runs = 0
    attempts = 0

    out_ep = "./Training Results/Chapter 3/Incremental/Rooms/Episode/Hybrid/"
    out_train = "./Training Results/Chapter 3/Incremental/Rooms/Train/Hybrid/"

    while successful_runs < target_successful_runs:
        attempts += 1
        print(f"\n[Hybrid] Attempt {attempts} (successful so far: {successful_runs}/{target_successful_runs})...")
        try:
            experiment_id = random.randrange(10000)

            env = XuFourRooms(movement_penalty=-0.01, goal_reward=1.0)
            test_env = XuFourRooms(movement_penalty=-0.01, goal_reward=1.0)

            agent = IncrementalHybridAgent(
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
                replace_drop_threshold=0.05,
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

            Path(out_ep).mkdir(parents=True, exist_ok=True)
            with open(f"{out_ep}/{experiment_id}-{uuid.uuid1()}.json", "w", encoding="utf-8") as f:
                json.dump(test_results, f, ensure_ascii=False, indent=4)

            Path(out_train).mkdir(parents=True, exist_ok=True)
            with open(f"{out_train}/{experiment_id}-{uuid.uuid1()}.json", "w", encoding="utf-8") as f:
                json.dump(train_results, f, ensure_ascii=False, indent=4)

            successful_runs += 1
            print(f"[Hybrid] Run {successful_runs} of {target_successful_runs} completed successfully.")

        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break

        except Exception as e:
            print(f"[Hybrid] Attempt {attempts} failed with an exception. Skipping this run.")
            print(f"Reason: {e}")
            traceback.print_exc()
            gc.collect()
            continue

    print(f"\n[Hybrid] Finished. Successful runs: {successful_runs}/{target_successful_runs}.")
