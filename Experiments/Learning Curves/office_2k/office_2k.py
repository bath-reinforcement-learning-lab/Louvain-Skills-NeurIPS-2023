# TODO: CAN WE EXTRACT THE "LENGTH" DATA FROM THE BINARY DATA?

import os

os.environ["OMP_NUM_THREADS"] = "2"  # export OMP_NUM_THREADS=2
os.environ["OPENBLAS_NUM_THREADS"] = "2"  # export OPENBLAS_NUM_THREADS=2
os.environ["MKL_NUM_THREADS"] = "2"  # export MKL_NUM_THREADS=2
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"  # export VECLIB_MAXIMUM_THREADS=2
os.environ["NUMEXPR_NUM_THREADS"] = "2"  # export NUMEXPR_NUM_THREADS=2

import random

from officeworld import OfficeWorldEnvironment
from officeworld.utils.serialisation import OfficeBuildingJSONHandler

from louvainskills.agent_trainers import (
    generate_aggregate_graphs,
    train_multi_level_agent,
    train_flat_agent,
    train_betweenness_agent,
    train_eigenoptions_agent,
    train_primitive_agent,
)

from louvainskills.louvain import apply_louvain
from louvainskills.label_propagation import apply_label_propagation
from louvainskills.node_betweenness import apply_node_betweenness
from louvainskills.eigenoptions import derive_pvfs

resolution = 0.05
epsilon = 0.1
alpha = 0.4
gamma = 1.0
default_action_value = 0.0
n_step_updates = True
num_agents = 20
test_interval = 5
num_epochs = 1000
epoch_length = 1000
test_episode_cutoff = 150
option_training_num_rollouts = 1
can_leave_initiation_set = False
results_directory = "./Training Results/Learning Curves/Office2k"


# Read office file and extract useful metadata.
office_name = "office_2k"
office = OfficeBuildingJSONHandler.load_from_json(f"./Experiments/Learning Curves/{office_name}/{office_name}.json")
office_rooms = office.rooms
num_floors = len(office.layout)

# Randomly choose a start and goal room.
start_floor = random.randint(0, num_floors - 1)
goal_floor = random.randint(0, num_floors - 1)
start_room = None
goal_room = None
while start_room == goal_room:
    start_room = random.choice(office_rooms[start_floor])
    goal_room = random.choice(office_rooms[goal_floor])

# Define environment arguments.
env_name = f"{office_name}"
explorable_kwargs = {
    "office": office,
    "start_floor": start_floor,
    "start_room": start_room,
    "explorable": True,
}
explorable_environment_args = (OfficeWorldEnvironment, explorable_kwargs, env_name)
kwargs = {
    "office": office,
    "start_floor": start_floor,
    "goal_floor": goal_floor,
    "start_room": start_room,
    "goal_room": goal_room,
    "movement_penalty": -0.01,
    "goal_reward": 1.0,
}
environment_args = (OfficeWorldEnvironment, kwargs, env_name)

for i in range(2):
    experiment_id = random.randrange(10000)

    aggregate_graphs, stg = generate_aggregate_graphs(
        explorable_environment_args,
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
        test_interval=test_interval,
        num_epochs=num_epochs,
        epoch_length=epoch_length,
        test_episode_cutoff=test_episode_cutoff,
        results_directory=results_directory,
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
        test_interval=test_interval,
        num_epochs=num_epochs,
        epoch_length=epoch_length,
        test_episode_cutoff=test_episode_cutoff,
        option_training_num_rollouts=option_training_num_rollouts,
        can_leave_initiation_set=can_leave_initiation_set,
        results_directory=results_directory,
        aggregate_graphs=aggregate_graphs,
        stg=stg,
        experiment_id=experiment_id,
    )

    # Eigenoptions
    pvfs, eig_stg = derive_pvfs(stg, 64)
    train_eigenoptions_agent(
        environment_args=environment_args,
        epsilon=epsilon,
        alpha=alpha,
        gamma=gamma,
        default_action_value=default_action_value,
        n_step_updates=n_step_updates,
        num_agents=num_agents,
        test_interval=test_interval,
        num_epochs=num_epochs,
        epoch_length=epoch_length,
        test_episode_cutoff=test_episode_cutoff,
        results_directory=results_directory,
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
        test_interval=test_interval,
        num_epochs=num_epochs,
        epoch_length=epoch_length,
        test_episode_cutoff=test_episode_cutoff,
        option_training_num_rollouts=option_training_num_rollouts,
        results_directory=results_directory,
        subgoals=subgoals,
        centralities=centralities,
        n_options=len(subgoals),
        initiation_set_size=128,
        stg=stg,
        experiment_id=experiment_id,
    )

    # Label Propagation Skills
    aggregate_graph, stg = generate_aggregate_graphs(explorable_environment_args, apply_label_propagation)
    train_flat_agent(
        environment_args=environment_args,
        epsilon=epsilon,
        alpha=alpha,
        gamma=gamma,
        default_action_value=default_action_value,
        n_step_updates=n_step_updates,
        num_agents=num_agents,
        test_interval=test_interval,
        num_epochs=num_epochs,
        epoch_length=epoch_length,
        test_episode_cutoff=test_episode_cutoff,
        option_training_num_rollouts=option_training_num_rollouts,
        can_leave_initiation_set=can_leave_initiation_set,
        results_directory=results_directory,
        aggregate_graphs=aggregate_graph,
        stg=stg,
        experiment_id=experiment_id,
    )
