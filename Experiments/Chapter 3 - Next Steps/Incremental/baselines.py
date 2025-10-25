import random

from simpleenvs.envs.discrete_rooms import XuFourRooms
from simpleenvs.envs.discrete_rooms.explorable_rooms import ExplorableXuFourRooms

from louvainskills.agent_trainers import (
    generate_aggregate_graphs,
    train_multi_level_agent,
    train_primitive_agent,
)
from louvainskills.louvain import apply_louvain


resolution = 0.05
epsilon = 0.1
alpha = 0.4
gamma = 1.0
default_action_value = 0.0
n_step_updates = True
num_agents = 10
test_interval = 1
num_epochs = 100
epoch_length = 100
test_episode_cutoff = 40
option_training_num_rollouts = 1
can_leave_initiation_set = False


M_REPEATS = 50


# Define explorable version of the environment and generate "clean" STG.
explorable_environment_args = (ExplorableXuFourRooms, {}, "Rooms")
env = explorable_environment_args[0](**explorable_environment_args[1])
env.reset()
clean_stg = env.generate_interaction_graph(directed=True, weighted=False)


for i in range(M_REPEATS):
    # Generate a random experiment ID.
    experiment_ids = [random.randrange(10000) for _ in range(M_REPEATS)]

    results_directory = "./Training Results/Chapter 3/Incremental/Rooms/"

    # Get Louvain partitions.
    louvain_aggregate_graphs, louvain_stg = generate_aggregate_graphs(
        explorable_environment_args,
        apply_louvain,
        {"resolution": resolution, "return_aggregate_graphs": True, "first_levels_to_skip": 1},
    )

    # Define target environment.
    env_name = "Rooms"
    kwargs = {
        "movement_penalty": -0.01,
        "goal_reward": 1.0,
    }
    environment_args = (XuFourRooms, kwargs, env_name)

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
        aggregate_graphs=louvain_aggregate_graphs,
        stg=louvain_stg,
        experiment_id=experiment_ids[i],
    )

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
        experiment_id=experiment_ids[i],
    )
