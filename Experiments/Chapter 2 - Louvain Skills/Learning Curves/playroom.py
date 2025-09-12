import random

from simpleenvs.envs.playroom import PlayroomEnvironment

from louvainskills.agent_trainers import (
    generate_aggregate_graphs,
    train_multi_level_agent,
    train_single_level_agents,
    train_flat_agent,
    train_xu_agent,
    train_betweenness_agent,
    train_eigenoptions_agent,
    train_primitive_agent,
)
from louvainskills.louvain import apply_louvain
from louvainskills.edge_betweenness import apply_edge_betweenness
from louvainskills.label_propagation import apply_label_propagation
from louvainskills.node_betweenness import apply_node_betweenness
from louvainskills.eigenoptions import derive_pvfs

resolution = 0.01
epsilon = 0.1
alpha = 0.2
gamma = 1.0
default_action_value = 0.0
n_step_updates = True
num_agents = 10
test_interval = 2
num_epochs = 1000
epoch_length = 100
test_episode_cutoff = 40
option_training_num_rollouts = 2500
can_leave_initiation_set = False


M_REPEATS = 49


# Define explorable version of the environment and generate "clean" STG.
explorable_environment_args = (
    PlayroomEnvironment,
    {
        "action_penalty": -0.01,
        "goal_reward": 1.0,
    },
    "Playroom",
)
env = explorable_environment_args[0](**explorable_environment_args[1])
env.reset()
clean_stg_weighted = env.generate_interaction_graph(directed=True, weighted=True)
clean_stg_unweighted = env.generate_interaction_graph(directed=True, weighted=False)

# Create a sequence of initial states to use across all agents.
# TODO: Consider moving inside the for loop to ensure different sequences?
initial_states = env.get_initial_states() * 5
random.shuffle(initial_states)

# Generate random experiment IDs.
experiment_ids = [random.randrange(10000) for _ in range(M_REPEATS)]

results_directory = "./Training Results/Chapter 2/Learning Curves/Playroom/"

for i in range(M_REPEATS):
    # Get Louvain partitions.
    louvain_aggregate_graphs, louvain_stg = generate_aggregate_graphs(
        explorable_environment_args,
        apply_louvain,
        {"resolution": resolution, "return_aggregate_graphs": True, "first_levels_to_skip": 1},
    )

    # Get node betweeness centralities.
    centralities, subgoals = apply_node_betweenness(clean_stg_unweighted.copy())

    # Get label propagation partition.
    lp_aggregate_graphs, lp_stg = generate_aggregate_graphs(explorable_environment_args, apply_label_propagation)

    # Get edge betweenness partition.
    eb_aggregate_graphs, eb_stg = generate_aggregate_graphs(explorable_environment_args, apply_edge_betweenness)

    # Get Louvain partition (for Xu et al. comparison).
    xu_aggregate_graphs, xu_stg = generate_aggregate_graphs(
        explorable_environment_args,
        apply_louvain,
        {"resolution": 1.0, "return_aggregate_graphs": True, "first_levels_to_skip": 0},
    )

    # Get PVFs for Eigenoptions.
    pvfs, eig_stg = derive_pvfs(clean_stg_unweighted.copy(), 16)

    # Define target environment.
    env_name = "Playroom"
    kwargs = {
        "action_penalty": -0.01,
        "goal_reward": 1.0,
        "initial_states_order": initial_states.copy(),
    }
    environment_args = (PlayroomEnvironment, kwargs, env_name)

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

    train_single_level_agents(
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
        aggregate_graphs=louvain_aggregate_graphs,
        stg=louvain_stg,
        experiment_id=experiment_ids[i],
        agent_type="Flat Louvain",
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
        experiment_id=experiment_ids[i],
    )

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
        initiation_set_size=30,
        stg=clean_stg_unweighted.copy(),
        experiment_id=experiment_ids[i],
    )

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
        aggregate_graphs=lp_aggregate_graphs,
        stg=lp_stg,
        experiment_id=experiment_ids[i],
        agent_type="Label Propagation",
    )

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
        aggregate_graphs=eb_aggregate_graphs,
        stg=eb_stg,
        experiment_id=experiment_ids[i],
        agent_type="Edge Betweenness",
    )

    train_xu_agent(
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
        aggregate_graphs=xu_aggregate_graphs,
        stg=xu_stg,
        experiment_id=experiment_ids[i],
    )

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
        experiment_id=experiment_ids[i],
    )
