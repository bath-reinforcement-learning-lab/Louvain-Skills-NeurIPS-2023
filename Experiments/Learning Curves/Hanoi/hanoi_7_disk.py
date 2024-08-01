import random

from simpleenvs.envs.hanoi import HanoiEnvironment
from simpleenvs.envs.hanoi.explorable_hanoi import ExplorableHanoiEnvironment

from louvainskills.agent_trainers import (
    generate_aggregate_graphs,
    train_multi_level_agent,
    train_single_level_agents,
    train_flat_agent,
    train_betweenness_agent,
    train_eigenoptions_agent,
    train_primitive_agent,
)

from louvainskills.louvain import apply_louvain
from louvainskills.edge_betweenness import apply_edge_betweenness
from louvainskills.label_propagation import apply_label_propagation
from louvainskills.node_betweenness import apply_node_betweenness
from louvainskills.eigenoptions import derive_pvfs

resolution = 0.05
epsilon = 0.1
alpha = 0.4
gamma = 1.0
default_action_value = 0.0
n_step_updates = True
num_agents = 3
num_epochs = 500
epoch_length = 1000
test_episode_cutoff = 2000
option_training_num_rollouts = 1
can_leave_initiation_set = False

left_to_right_env_args = (
    HanoiEnvironment,
    {"num_disks": 7, "num_poles": 3, "start_state": (0, 0, 0, 0, 0, 0, 0), "goal_state": (2, 2, 2, 2, 2, 2, 2)},
    "Hanoi3P7D",
)
left_to_mid_env_args = (
    HanoiEnvironment,
    {"num_disks": 7, "num_poles": 3, "start_state": (0, 0, 0, 0, 0, 0, 0), "goal_state": (1, 1, 1, 1, 1, 1, 1)},
    "Hanoi3P7D",
)
mid_to_right_env_args = (
    HanoiEnvironment,
    {"num_disks": 7, "num_poles": 3, "start_state": (1, 1, 1, 1, 1, 1, 1), "goal_state": (2, 2, 2, 2, 2, 2, 2)},
    "Hanoi3P7D",
)
right_to_left_env_args = (
    HanoiEnvironment,
    {"num_disks": 7, "num_poles": 3, "start_state": (2, 2, 2, 2, 2, 2, 2), "goal_state": (0, 0, 0, 0, 0, 0, 0)},
    "Hanoi3P7D",
)

target_envs_args_list = [
    left_to_right_env_args,
    left_to_mid_env_args,
    mid_to_right_env_args,
    right_to_left_env_args,
]

for j, target_env_args in enumerate(target_envs_args_list):
    for i in range(1):
        # Define goal environment that we're deriving skills from.
        experiment_id = random.randrange(10000)
        GoalEnvironmentType = ExplorableHanoiEnvironment
        goal_env_name = "Hanoi"
        goal_kwargs = {"num_disks": 7, "num_poles": 3, "start_state": (0, 0, 0, 0, 0, 0, 0)}
        goal_environment_args = (GoalEnvironmentType, goal_kwargs, goal_env_name)

        # Define target environment.
        env_type = target_env_args[0]
        kwargs = target_env_args[1]
        env_name = target_env_args[2]
        environment_args = (env_type, kwargs, env_name)

        aggregate_graphs, stg = generate_aggregate_graphs(
            goal_environment_args,
            apply_louvain,
            {"resolution": resolution, "return_aggregate_graphs": True, "first_levels_to_skip": 1},
        )

        # Q-Learning with Primitives
        train_primitive_agent(
            environment_args=environment_args,
            epsilon=epsilon,
            alpha=alpha,
            gamma=gamma,
            default_action_value=default_action_value,
            num_agents=num_agents,
            num_epochs=num_epochs,
            epoch_length=epoch_length,
            test_episode_cutoff=test_episode_cutoff,
            output_directory="Primitive Agent",
            experiment_id=experiment_id,
        )

        # Multi-Level Louvain Skills
        train_multi_level_agent(
            environment_args=environment_args,
            epsilon=epsilon,
            alpha=alpha,
            gamma=gamma,
            default_action_value=default_action_value,
            n_step_updates=n_step_updates,
            num_agents=num_agents,
            num_epochs=num_epochs,
            epoch_length=epoch_length,
            test_episode_cutoff=test_episode_cutoff,
            option_training_num_rollouts=option_training_num_rollouts,
            can_leave_initiation_set=can_leave_initiation_set,
            output_directory="Multi-Level Agent",
            aggregate_graphs=aggregate_graphs,
            stg=stg,
            experiment_id=experiment_id,
        )

        # Eigenoptions
        pvfs, eig_stg = derive_pvfs(stg, 16)
        train_eigenoptions_agent(
            environment_args=environment_args,
            epsilon=epsilon,
            alpha=alpha,
            gamma=gamma,
            default_action_value=default_action_value,
            n_step_updates=n_step_updates,
            num_agents=num_agents,
            num_epochs=num_epochs,
            epoch_length=epoch_length,
            test_episode_cutoff=test_episode_cutoff,
            output_directory="Eigenoptions",
            pvfs=pvfs,
            stg=eig_stg,
            experiment_id=experiment_id,
        )

        # Node Betweenness Subgoal Skills
        centralities, subgoals = apply_node_betweenness(stg)
        train_betweenness_agent(
            environment_args=environment_args,
            epsilon=epsilon,
            alpha=alpha,
            gamma=gamma,
            default_action_value=default_action_value,
            n_step_updates=n_step_updates,
            num_agents=num_agents,
            num_epochs=num_epochs,
            epoch_length=epoch_length,
            test_episode_cutoff=test_episode_cutoff,
            option_training_num_rollouts=option_training_num_rollouts,
            output_directory="Betweenness",
            subgoals=subgoals,
            centralities=centralities,
            n_options=len(subgoals),
            initiation_set_size=30,
            stg=stg,
            experiment_id=experiment_id,
        )

        # Label Propagation Skills
        aggregate_graph, stg = generate_aggregate_graphs(goal_environment_args, apply_label_propagation)
        train_flat_agent(
            environment_args=environment_args,
            epsilon=epsilon,
            alpha=alpha,
            gamma=gamma,
            default_action_value=default_action_value,
            n_step_updates=n_step_updates,
            num_agents=num_agents,
            num_epochs=num_epochs,
            epoch_length=epoch_length,
            test_episode_cutoff=test_episode_cutoff,
            option_training_num_rollouts=option_training_num_rollouts,
            can_leave_initiation_set=can_leave_initiation_set,
            output_directory="Label Propagation",
            aggregate_graphs=aggregate_graph,
            stg=stg,
            experiment_id=experiment_id,
        )

        # Edge Betweenness Skills
        aggregate_graph, stg = generate_aggregate_graphs(goal_environment_args, apply_edge_betweenness)
        train_flat_agent(
            environment_args=environment_args,
            epsilon=epsilon,
            alpha=alpha,
            gamma=gamma,
            default_action_value=default_action_value,
            n_step_updates=n_step_updates,
            num_agents=num_agents,
            num_epochs=num_epochs,
            epoch_length=epoch_length,
            test_episode_cutoff=test_episode_cutoff,
            option_training_num_rollouts=option_training_num_rollouts,
            can_leave_initiation_set=can_leave_initiation_set,
            output_directory="Edge Betweenness",
            aggregate_graphs=aggregate_graph,
            stg=stg,
            experiment_id=experiment_id,
        )
