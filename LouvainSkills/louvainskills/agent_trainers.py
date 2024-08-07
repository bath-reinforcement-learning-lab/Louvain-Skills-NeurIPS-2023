import gc
import json
import uuid
import copy

import networkx as nx

import louvainskills.utils.istarmap

from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

from louvainskills.options import LouvainOption
from louvainskills.options import SubgoalOption
from louvainskills.options import EigenOption
from louvainskills.utils.graph_utils import convert_nx_to_ig, convert_ig_to_nx
from louvainskills.option_trainers import ValueIterationOptionTrainer

from simpleoptions import BaseOption, PrimitiveOption, OptionAgent

from typing import Type, Dict, Tuple, List, Hashable, Mapping


def generate_aggregate_graphs(
    environment_args, clustering_method, clustering_method_args={}, directed=True, weighted=False
):
    (EnvironmentType, kwargs, env_name) = environment_args

    # Initialise environment.
    env = EnvironmentType(**kwargs)
    env.reset()

    # Generate networkx state-transition diagram.
    if not weighted:
        stg = env.generate_interaction_graph(directed=directed)
    else:
        stg = env.generate_interaction_graph(directed=True, weighted=True)

    # Convert networkx to igraph.
    stg_ig = convert_nx_to_ig(stg)

    # Perform hierarchical graph clustering.
    stg_ig, aggregate_graphs_ig = clustering_method(stg_ig, **clustering_method_args)

    # Convert igraph stg to networkx.
    stg = convert_ig_to_nx(stg_ig)

    # Convert igraph aggregate graphs to networkx.
    aggregate_graphs = []
    for i, aggregate_graph_ig in enumerate(aggregate_graphs_ig):
        # print(f"\nLevel {i}")
        aggregate_graph = convert_ig_to_nx(aggregate_graph_ig)

        if aggregate_graph.number_of_nodes() > 1:
            aggregate_graphs.append(copy.deepcopy(aggregate_graph))

    return aggregate_graphs, stg


def train_multi_level_agent(
    environment_args: Tuple[Type, Dict, str],
    epsilon: float,
    alpha: float,
    gamma: float,
    default_action_value: float,
    n_step_updates: bool,
    num_agents: int,
    test_interval: int,
    num_epochs: int,
    epoch_length: int,
    test_episode_cutoff: int,
    option_training_num_rollouts: int,
    can_leave_initiation_set: bool,
    output_directory: str,
    aggregate_graphs: List[nx.DiGraph],
    stg: nx.DiGraph,
    experiment_id: int,
):
    """
    Generate a multi-level hierarchy of Louvain options and use them to train a Macro-Q/Intra-Option Learning agent.

    Args:
        environment_args (Tuple[Type, Dict, str]): The type, arguments, and name of the environment to train the agent in.
        epsilon (float): Exploration rate for the agent.
        alpha (float): Learning rate for the agent.
        gamma (float): Discount factor for the agent.
        default_action_value (float): The default value to assign to unseen state-action pairs.
        n_step_updates (bool): Whether or not to use n-step updates.
        num_agents (int): The number of agents to train.
        test_interval (int): The interval at which to test the agent, in epochs.
        num_epochs (int): The number of epochs to train the agent for.
        epoch_length (int): The length of each epoch, in primitive decision stages.
        test_episode_cutoff (int): The number of primitive decision stages after which a test episode is cut off.
        option_training_num_rollouts (int): The number of rollouts to use when training options.
        can_leave_initiation_set (int): Whether or not agents executing a Louvian option are allowed to leave the initiation set.
        output_directory (str): The directory to save the training results in.
        aggregate_graphs (List[nx.DiGraph]): A list of aggregate graphs representing the hierarchy of skills to train.
        stg (nx.DiGraph): The state-transition graph of the environment.
        experiment_id (int): The ID of the experiment being run.
    """
    (EnvironmentType, kwargs, env_name) = environment_args

    # Initialise environment.
    env = EnvironmentType(**kwargs)
    env.reset()

    # Create a list of skills to train.
    skill_hierarchy = []
    for i, aggregate_graph in enumerate(aggregate_graphs[1:]):
        skill_hierarchy.append([])
        print(f"\nHierarchy Level {i}")
        for u, v in aggregate_graph.edges():
            if u != v:
                print(f"({i}: {u} --> {v})")
                skill_hierarchy[i].append((i, u, v))

    # Create options representing primitive actions.
    primitive_options = []
    for action in env.get_action_space():
        primitive_options.append(PrimitiveOption(action, env))
    env.set_options(primitive_options)

    # Train higher-level options.
    if option_training_num_rollouts == 1:
        option_trainer = ValueIterationOptionTrainer(env, stg, gamma=1.0, theta=1e-5, deterministic=True)
    else:
        option_trainer = ValueIterationOptionTrainer(
            env, stg, gamma=1.0, theta=1e-5, num_rollouts=option_training_num_rollouts
        )

    options = []
    for level, hierarchy_level in tqdm(enumerate(skill_hierarchy), desc=f"Hierachy Level"):
        options.append([])

        # Set available options to options from the previous level of the hierarchy.
        if level == 0:
            env.set_options(primitive_options)
        else:
            env.set_options(options[level - 1])

        # Train this level of the hierarchy.
        for i, u, v in tqdm(hierarchy_level, desc="Training Skills"):
            # Train skills at this level of the hierarchy.
            options[level].append(
                option_trainer.train_option_policy(
                    LouvainOption(stg, i, u, v, can_leave_initiation_set), can_leave_initiation_set
                )
            )
            # For Debugging - Saves the STG.
            nx.write_gexf(stg, f"{env_name} Policy Labelled.gexf", prettyprint=True)
            # quit()

    # For Debugging - Saves the STG.
    nx.write_gexf(stg, f"{env_name} Policy Labelled.gexf", prettyprint=True)
    # quit()

    options = [option for level in options for option in level]
    options.extend(primitive_options)
    gc.collect()

    # Generate results.
    # Run Macro-Q Learning Agent
    for run in tqdm(range(num_agents), desc="Multi-Level Agent"):
        # Initialise our environment.
        env = EnvironmentType(**kwargs)
        env.set_options(options)
        test_env = EnvironmentType(**kwargs)
        test_env.set_options(options)

        # Initialise our agent and train it.
        agent = OptionAgent(
            env,
            test_env=test_env,
            epsilon=epsilon,
            macro_alpha=alpha,
            intra_option_alpha=alpha,
            gamma=gamma,
            n_step_updates=n_step_updates,
            default_action_value=default_action_value,
        )

        train_results, episode_test_results = agent.run_agent(
            num_epochs=num_epochs,
            epoch_length=epoch_length,
            test_interval=test_interval,
            test_length=test_episode_cutoff,
            test_runs=5,
            verbose_logging=False,
            epoch_eval=False,
            episodic_eval=True,
        )

        run_id = uuid.uuid1()

        # Save training performance.
        train_dir = f"./Training Results/Learning Curves/{env_name}/Train/{output_directory}/"
        Path(train_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{train_dir}/{experiment_id}-{run}-{run_id}.json", "w", encoding="utf-8") as f:
            json.dump(train_results, f, ensure_ascii=False, indent=4)

        # Save epoch-based evaluation performance.
        # epoch_test_dir = f"./Training Results/Learning Curves/{env_name}/Epoch/{output_directory}/"
        # Path(epoch_test_dir).mkdir(parents=True, exist_ok=True)
        # with open(f"{epoch_test_dir}/{experiment_id}-{run}-{run_id}.json", "w", encoding="utf-8") as f:
        #     json.dump(epoch_test_results, f, ensure_ascii=False, indent=4)

        # Save episode-based evaluation performance.
        episode_test_dir = f"./Training Results/Learning Curves/{env_name}/Episode/{output_directory}/"
        Path(episode_test_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{episode_test_dir}/{experiment_id}-{run}-{run_id}.json", "w", encoding="utf-8") as f:
            json.dump(episode_test_results, f, ensure_ascii=False, indent=4)

        gc.collect()


def train_single_level_agents(
    environment_args: Tuple[Type, Dict, str],
    epsilon: float,
    alpha: float,
    gamma: float,
    default_action_value: float,
    n_step_updates: bool,
    num_agents: int,
    test_interval: int,
    num_epochs: int,
    epoch_length: int,
    test_episode_cutoff: int,
    option_training_num_rollouts: int,
    can_leave_initiation_set: bool,
    output_directory: str,
    aggregate_graphs: List[nx.DiGraph],
    stg: nx.DiGraph,
    experiment_id: int,
):
    """
    Generate a series of single-level skill hierarchies, each based on a partition of the state-transition graph determined
    by a single level of the clsuter hierarchy produced by the Louvain algorithm.
    Then, train Macro-Q/Intra-Option Learning agents using these hierarchies.

    Args:
        environment_args (Tuple[Type, Dict, str]): The type, arguments, and name of the environment to train the agent in.
        epsilon (float): Exploration rate for the agent.
        alpha (float): Learning rate for the agent.
        gamma (float): Discount factor for the agent.
        default_action_value (float): The default value to assign to unseen state-action pairs.
        n_step_updates (bool): Whether or not to use n-step updates.
        num_agents (int): The number of agents to train.
        test_interval (int): The interval at which to test the agent, in epochs.
        num_epochs (int): The number of epochs to train the agent for.
        epoch_length (int): The length of each epoch, in primitive decision stages.
        test_episode_cutoff (int): The number of primitive decision stages after which a test episode is cut off.
        option_training_num_rollouts (int): The number of rollouts to use when training options.
        can_leave_initiation_set (bool): Whether or not agents executing a Louvian option are allowed to leave the initiation set.
        output_directory (str): The directory to save the training results in.
        aggregate_graphs (List[nx.DiGraph]): A list of aggregate graphs representing the hierarchy of skills to train.
        stg (nx.DiGraph): The state-transition graph of the environment.
        experiment_id (int): The ID of the experiment being run.
    """
    (EnvironmentType, kwargs, env_name) = environment_args

    # Initialise environment.
    env = EnvironmentType(**kwargs)
    env.reset()

    # Create a list of skills to train.
    skill_hierarchy = []
    for i, aggregate_graph in enumerate(aggregate_graphs[1:]):
        skill_hierarchy.append([])
        print(f"\nHierarchy Level {i}")
        for u, v in aggregate_graph.edges():
            if u != v:
                print(f"({i}: {u} --> {v})")
                skill_hierarchy[i].append((i, u, v))

    # Create options representing primitive actions.
    primitive_options = []
    for action in env.get_action_space():
        primitive_options.append(PrimitiveOption(action, env))
    env.set_options(primitive_options)

    # Train higher-level options.
    if option_training_num_rollouts == 1:
        option_trainer = ValueIterationOptionTrainer(env, stg, gamma=1.0, theta=1e-5, deterministic=True)
    else:
        option_trainer = ValueIterationOptionTrainer(
            env, stg, gamma=1.0, theta=1e-5, num_rollouts=option_training_num_rollouts
        )

    options = []
    for level, hierarchy_level in tqdm(enumerate(skill_hierarchy), desc=f"Hierachy Level"):
        options.append([])
        for i, u, v in tqdm(hierarchy_level, desc="Training Skills"):
            options[level].append(
                option_trainer.train_option_policy(
                    LouvainOption(stg, i, u, v, can_leave_initiation_set), can_leave_initiation_set
                )
            )
    gc.collect()

    # For Debugging - Saves the STG.
    # nx.write_gexf(stg, f"{env_name} Policy Labelled.gexf", prettyprint=True)
    # quit()

    # Generate results.
    # Run Macro-Q Learning Agents.
    for level, level_options in enumerate(options):
        for run in tqdm(range(num_agents), desc=f"Level {level} Agent"):
            # Initialise our environment.
            env = EnvironmentType(**kwargs)
            env.set_options(primitive_options)
            env.set_options(level_options, append=True)

            test_env = EnvironmentType(**kwargs)
            test_env.set_options(primitive_options)
            test_env.set_options(level_options, append=True)

            # Initialise our agent and train it.
            agent = OptionAgent(
                env,
                test_env=test_env,
                epsilon=epsilon,
                macro_alpha=alpha,
                intra_option_alpha=alpha,
                gamma=gamma,
                n_step_updates=n_step_updates,
                default_action_value=default_action_value,
            )
            train_results, episode_test_results = agent.run_agent(
                num_epochs=num_epochs,
                epoch_length=epoch_length,
                test_interval=test_interval,
                test_length=test_episode_cutoff,
                test_runs=5,
                verbose_logging=False,
                epoch_eval=False,
                episodic_eval=True,
            )

            run_id = uuid.uuid1()

            # Save training performance.
            train_dir = f"./Training Results/Learning Curves/{env_name}/Train/{output_directory}/Level {level}/"
            Path(train_dir).mkdir(parents=True, exist_ok=True)
            with open(f"{train_dir}/{experiment_id}-{run}-{run_id}.json", "w", encoding="utf-8") as f:
                json.dump(train_results, f, ensure_ascii=False, indent=4)

            # Save epoch-based evaluation performance.
            # epoch_test_dir = f"./Training Results/Learning Curves/{env_name}/Epoch/{output_directory}/Level {level}/"
            # Path(epoch_test_dir).mkdir(parents=True, exist_ok=True)
            # with open(f"{epoch_test_dir}/{experiment_id}-{run}-{run_id}.json", "w", encoding="utf-8") as f:
            #     json.dump(epoch_test_results, f, ensure_ascii=False, indent=4)

            # Save epoch-based evaluation performance.
            episode_test_dir = (
                f"./Training Results/Learning Curves/{env_name}/Episode/{output_directory}/Level {level}/"
            )
            Path(episode_test_dir).mkdir(parents=True, exist_ok=True)
            with open(f"{episode_test_dir}/{experiment_id}-{run}-{run_id}.json", "w", encoding="utf-8") as f:
                json.dump(episode_test_results, f, ensure_ascii=False, indent=4)

            gc.collect()


def train_flat_agent(
    environment_args: Tuple[Type, Dict, str],
    epsilon: float,
    alpha: float,
    gamma: float,
    default_action_value: float,
    n_step_updates: bool,
    num_agents: int,
    test_interval: int,
    num_epochs: int,
    epoch_length: int,
    test_episode_cutoff: int,
    option_training_num_rollouts: int,
    can_leave_initiation_set: bool,
    output_directory: str,
    aggregate_graphs: List[nx.DiGraph],
    stg: nx.DiGraph,
    experiment_id: int,
):
    """
    Generate a single-level hierarchy of Louvain options and use them to train a Macro-Q/Intra-Option Learning agent.
    The hierarchy is based on the full set of partitions produced by the Louvain algorithm, but is flat (i.e., all option
    policies are defined over primitive actions). Then, train a Macro-Q/Intra-Option Learning agent using these options.

    Args:
        environment_args (Tuple[Type, Dict, str]): The type, arguments, and name of the environment to train the agent in.
        epsilon (float): Exploration rate for the agent.
        alpha (float): Learning rate for the agent.
        gamma (float): Discount factor for the agent.
        default_action_value (float): The default value to assign to unseen state-action pairs.
        n_step_updates (bool): Whether or not to use n-step updates.
        num_agents (int): The number of agents to train.
        test_interval (int): The interval at which to test the agent, in epochs.
        num_epochs (int): The number of epochs to train the agent for.
        epoch_length (int): The length of each epoch, in primitive decision stages.
        test_episode_cutoff (int): The number of primitive decision stages after which a test episode is cut off.
        option_training_num_rollouts (int): The number of rollouts to use when training options.
        can_leave_initiation_set (bool): Whether or not agents executing a Louvian option are allowed to leave the initiation set.
        output_directory (str): The directory to save the training results in.
        aggregate_graphs (List[nx.DiGraph]): A list of aggregate graphs representing the hierarchy of skills to train.
        stg (nx.DiGraph): The state-transition graph of the environment.
        experiment_id (int): The ID of the experiment being run.
    """

    (EnvironmentType, kwargs, env_name) = environment_args

    # Initialise environment.
    env = EnvironmentType(**kwargs)
    env.reset()

    # Create a list of skills to train.
    skill_hierarchy = []
    for i, aggregate_graph in enumerate(aggregate_graphs[1:] if len(aggregate_graphs) > 1 else aggregate_graphs):
        skill_hierarchy.append([])
        print(f"\nHierarchy Level {i}")
        for u, v in aggregate_graph.edges():
            if u != v:
                print(f"({i}: {u} --> {v})")
                skill_hierarchy[i].append((i, u, v))

    # Create options representing primitive actions.
    primitive_options = []
    for action in env.get_action_space():
        primitive_options.append(PrimitiveOption(action, env))
    env.set_options(primitive_options)

    # Train higher-level options.
    if option_training_num_rollouts == 1:
        option_trainer = ValueIterationOptionTrainer(env, stg, gamma=1.0, theta=1e-5, deterministic=True)
    else:
        option_trainer = ValueIterationOptionTrainer(
            env, stg, gamma=1.0, theta=1e-5, num_rollouts=option_training_num_rollouts
        )

    options = []
    for hierarchy_level in tqdm(skill_hierarchy, desc=f"Hierachy Level"):
        for i, u, v in tqdm(hierarchy_level, desc="Training Skills"):
            options.append(
                option_trainer.train_option_policy(
                    LouvainOption(stg, i, u, v, can_leave_initiation_set), can_leave_initiation_set
                )
            )
    options.extend(primitive_options)
    gc.collect()

    # For Debugging - Saves the STG.
    # nx.write_gexf(stg, f"{env_name} Policy Labelled.gexf", prettyprint=True)
    # quit()

    # Generate results.
    # Run Macro-Q Learning Agent
    for run in tqdm(range(num_agents), desc="Flat Agent"):
        # Initialise our environment.
        env = EnvironmentType(**kwargs)
        env.set_options(options)

        test_env = EnvironmentType(**kwargs)
        test_env.set_options(options)

        # Initialise our agent and train it.
        agent = OptionAgent(
            env,
            test_env=test_env,
            epsilon=epsilon,
            macro_alpha=alpha,
            intra_option_alpha=alpha,
            gamma=gamma,
            n_step_updates=n_step_updates,
            default_action_value=default_action_value,
        )
        train_results, episode_test_results = agent.run_agent(
            num_epochs=num_epochs,
            epoch_length=epoch_length,
            test_interval=test_interval,
            test_length=test_episode_cutoff,
            test_runs=5,
            verbose_logging=False,
            epoch_eval=False,
            episodic_eval=True,
        )

        run_id = uuid.uuid1()

        # Save training performance.
        train_dir = f"./Training Results/Learning Curves/{env_name}/Train/{output_directory}/"
        Path(train_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{train_dir}/{experiment_id}-{run}-{uuid.uuid1()}.json", "w", encoding="utf-8") as f:
            json.dump(train_results, f, ensure_ascii=False, indent=4)

        # Save epoch-based evaluation performance.
        # epoch_test_dir = f"./Training Results/Learning Curves/{env_name}/Epoch/{output_directory}/"
        # Path(epoch_test_dir).mkdir(parents=True, exist_ok=True)
        # with open(f"{epoch_test_dir}/{experiment_id}-{run}-{run_id}.json", "w", encoding="utf-8") as f:
        #     json.dump(epoch_test_results, f, ensure_ascii=False, indent=4)

        # Save episode-based evaluation performance.
        episode_test_dir = f"./Training Results/Learning Curves/{env_name}/Episode/{output_directory}/"
        Path(episode_test_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{episode_test_dir}/{experiment_id}-{run}-{run_id}.json", "w", encoding="utf-8") as f:
            json.dump(episode_test_results, f, ensure_ascii=False, indent=4)

        gc.collect()
            num_epochs=num_epochs,
            epoch_length=epoch_length,
            test_interval=test_interval,
            test_length=test_episode_cutoff,
            test_runs=5,
            verbose_logging=False,
            epoch_eval=False,
            episodic_eval=True,
        )

        run_id = uuid.uuid1()

        # Save training performance.
        train_dir = f"./Training Results/Learning Curves/{env_name}/Train/{output_directory}/"
        Path(train_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{train_dir}/{experiment_id}-{run}-{uuid.uuid1()}.json", "w", encoding="utf-8") as f:
            json.dump(train_results, f, ensure_ascii=False, indent=4)

        # Save epoch-based evaluation performance.
        # epoch_test_dir = f"./Training Results/Learning Curves/{env_name}/Epoch/{output_directory}/"
        # Path(epoch_test_dir).mkdir(parents=True, exist_ok=True)
        # with open(f"{epoch_test_dir}/{experiment_id}-{run}-{run_id}.json", "w", encoding="utf-8") as f:
        #     json.dump(epoch_test_results, f, ensure_ascii=False, indent=4)

        # Save episode-based evaluation performance.
        episode_test_dir = f"./Training Results/Learning Curves/{env_name}/Episode/{output_directory}/"
        Path(episode_test_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{episode_test_dir}/{experiment_id}-{run}-{run_id}.json", "w", encoding="utf-8") as f:
            json.dump(episode_test_results, f, ensure_ascii=False, indent=4)

        gc.collect()


def train_betweenness_agent(
    environment_args: Tuple[Type, Dict, str],
    epsilon: float,
    alpha: float,
    gamma: float,
    default_action_value: float,
    n_step_updates: bool,
    num_agents: int,
    test_interval: int,
    num_epochs: int,
    epoch_length: int,
    test_episode_cutoff: int,
    option_training_num_rollouts: int,
    output_directory: str,
    subgoals: List[Hashable],
    centralities: Mapping[Hashable, float],
    n_options: int,
    initiation_set_size: int,
    stg: nx.DiGraph,
    experiment_id: int,
):
    """
    Generate a set of skills for navigating to the n highest local maxima of betweenness on the state-transition graph.
    Then, train a Macro-Q/Intra-Option Learning agent using these options.

    Args:
        environment_args (Tuple[Type, Dict, str]): The type, arguments, and name of the environment to train the agent in.
        epsilon (float): Exploration rate for the agent.
        alpha (float): Learning rate for the agent.
        gamma (float): Discount factor for the agent.
        default_action_value (float): The default value to assign to unseen state-action pairs.
        n_step_updates (bool): Whether or not to use n-step updates.
        num_agents (int): The number of agents to train.
        test_interval (int): The interval at which to test the agent, in epochs.
        num_epochs (int): The number of epochs to train the agent for.
        epoch_length (int): The length of each epoch, in primitive decision stages.
        test_episode_cutoff (int): The number of primitive decision stages after which a test episode is cut off.
        option_training_num_rollouts (int): The number of rollouts to use when training options.
        output_directory (str): The directory to save the training results in.
        subgoals (List[Hashable]): The subgoals to train options for.
        centralities (Mapping[Hashable, float]): The betweenness centralities of each node in the state-transition graph.
        n_options (int): The number of options to train.
        initiation_set_size (int): The size of the initiation set for each option.
        stg (nx.DiGraph): The state-transition graph of the environment.
        experiment_id (int): The ID of the experiment being run.
    """
    (EnvironmentType, kwargs, env_name) = environment_args

    # Initialise environment.
    env = EnvironmentType(**kwargs)
    env.reset()

    # Create options representing primitive actions.
    primitive_options = []
    for action in env.get_action_space():
        primitive_options.append(PrimitiveOption(action, env))
    env.set_options(primitive_options)

    # List n highest local maxima of betweenness, where n = n_options.
    subgoals = [(subgoal, centralities[subgoal]) for subgoal in subgoals]
    subgoals.sort(key=lambda x: x[1], reverse=True)
    subgoals = list(list(zip(*subgoals))[0])[:n_options]

    print(f"Subgoals: {subgoals}")

    # Train higher-level options.
    if option_training_num_rollouts == 1:
        option_trainer = ValueIterationOptionTrainer(env, stg, gamma=1.0, theta=1e-5, deterministic=True)
    else:
        option_trainer = ValueIterationOptionTrainer(
            env, stg, gamma=1.0, theta=1e-5, num_rollouts=option_training_num_rollouts
        )

    options = []
    option_trainer = ValueIterationOptionTrainer(env, stg, gamma=1.0, theta=1e-5, deterministic=True)
    for subgoal in tqdm(subgoals, desc="Betweenness Options"):
        # Define initiation set as the n closest nodes that have a path to the subgoal node, where n = initiation_set_size.
        initiation_set = sorted(list(nx.single_target_shortest_path_length(stg, subgoal)), key=lambda x: x[1])
        initiation_set = list(list(zip(*initiation_set))[0])[1 : min(initiation_set_size + 1, len(initiation_set) - 1)]
        options.append(option_trainer.train_option_policy(SubgoalOption(stg, subgoal, initiation_set), False))

    options.extend(primitive_options)
    gc.collect()

    # For Debugging - Saves the STG.
    nx.write_gexf(stg, f"{env_name} Policy Labelled.gexf", prettyprint=True)
    # quit()

    # Generate results.
    # Run Macro-Q Learning Agent
    for run in tqdm(range(num_agents), desc="Betweenness Agent"):
        # Initialise our environment.
        env = EnvironmentType(**kwargs)
        env.set_options(options)

        test_env = EnvironmentType(**kwargs)
        test_env.set_options(options)

        # Initialise our agent and train it.
        agent = OptionAgent(
            env,
            test_env=test_env,
            epsilon=epsilon,
            macro_alpha=alpha,
            intra_option_alpha=alpha,
            gamma=gamma,
            n_step_updates=n_step_updates,
            default_action_value=default_action_value,
        )
        train_results, episode_test_results = agent.run_agent(
            num_epochs=num_epochs,
            epoch_length=epoch_length,
            test_interval=test_interval,
            test_length=test_episode_cutoff,
            test_runs=5,
            verbose_logging=False,
            epoch_eval=False,
            episodic_eval=True,
        )

        run_id = uuid.uuid1()

        # Save training performance.
        train_dir = f"./Training Results/Learning Curves/{env_name}/Train/{output_directory}/"
        Path(train_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{train_dir}/{experiment_id}-{run}-{run_id}.json", "w", encoding="utf-8") as f:
            json.dump(train_results, f, ensure_ascii=False, indent=4)

        # Save epoch-based evaluation performance.
        # epoch_test_dir = f"./Training Results/Learning Curves/{env_name}/Epoch/{output_directory}/"
        # Path(epoch_test_dir).mkdir(parents=True, exist_ok=True)
        # with open(f"{epoch_test_dir}/{experiment_id}-{run}-{run_id}.json", "w", encoding="utf-8") as f:
        #     json.dump(epoch_test_results, f, ensure_ascii=False, indent=4)

        # Save episode-based evaluation performance.
        episode_test_dir = f"./Training Results/Learning Curves/{env_name}/Episode/{output_directory}/"
        Path(episode_test_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{episode_test_dir}/{experiment_id}-{run}-{run_id}.json", "w", encoding="utf-8") as f:
            json.dump(episode_test_results, f, ensure_ascii=False, indent=4)

        gc.collect()


def train_eigenoptions_agent(
    environment_args: Tuple[Type, Dict, str],
    epsilon: float,
    alpha: float,
    gamma: float,
    default_action_value: float,
    n_step_updates: bool,
    num_agents: int,
    test_interval: int,
    num_epochs: int,
    epoch_length: int,
    test_episode_cutoff: int,
    output_directory: str,
    pvfs: List[Dict[Hashable, float]],
    stg: nx.DiGraph,
    experiment_id: int,
    training_env_args: Tuple[Type, Dict, str] = None,
):
    """
    Generate a set of skills for "traversing the principal directions of the environment", based on the Eigenoptions
    method proposed by Machado et al. (2017). Then, train a Macro-Q/Intra-Option Learning agent using these options.

    Args:
        environment_args (Tuple[Type, Dict, str]): The type, arguments, and name of the environment to train the agent in.
        epsilon (float): Exploration rate for the agent.
        alpha (float): Learning rate for the agent.
        gamma (float): Discount factor for the agent.
        default_action_value (float): The default value to assign to unseen state-action pairs.
        n_step_updates (bool): Whether or not to use n-step updates.
        num_agents (int): The number of agents to train.
        test_interval (int): The interval at which to test the agent, in epochs.
        num_epochs (int): The number of epochs to train the agent for.
        epoch_length (int): The length of each epoch, in primitive decision stages.
        test_episode_cutoff (int): The number of primitive decision stages after which a test episode is cut off.
        output_directory (str): The directory to save the training results in.
        pvfs (List[Dict[Hashable, float]]): The proto-value functions to train options based on.
        stg (nx.DiGraph): The state-transition graph of the environment.
        experiment_id (int): The ID of the experiment being run.
        training_env_args (Tuple[Type, Dict, str], optional): The type, arguments, and name of the environment to train the eigenoptions in. Defaults to None, in which case the same environment is used for training and
    """
    (EnvironmentType, kwargs, env_name) = environment_args

    # Initialise environment.
    env = EnvironmentType(**kwargs)
    env.reset()

    # Create options representing primitive actions.
    primitive_options = []
    for action in env.get_action_space():
        primitive_options.append(PrimitiveOption(action, env))
    env.set_options(primitive_options)

    # Create one Eigenoption for each PVF.
    eigenoptions = []
    if training_env_args is None:
        for i, pvf in tqdm(enumerate(pvfs), desc="Eigenoptions"):
            eigenoptions.append(EigenOption(env, stg, pvf, i))
            eigenoptions[i].train()
    else:
        (TrainingEnvType, training_kwargs, training_env_name) = training_env_args
        training_env = TrainingEnvType(**training_kwargs)
        training_env.reset()

        # Create options representing primitive actions.
        training_primitive_options = []
        for action in training_env.get_action_space():
            training_primitive_options.append(PrimitiveOption(action, training_env))
        training_env.set_options(training_primitive_options)

        for i, pvf in tqdm(enumerate(pvfs), desc="Eigenoptions"):
            eigenoptions.append(EigenOption(training_env, stg, pvf, i))
            eigenoptions[i].train()
    gc.collect()

    # Generate results.
    # Run Macro-Q Learning Agent
    for run in tqdm(range(num_agents), desc="Eigenoptions Agent"):
        # Initialise our environment.
        env = EnvironmentType(**kwargs)
        env.set_options(primitive_options)
        env.set_exploration_options(eigenoptions)

        test_env = EnvironmentType(**kwargs)
        test_env.set_options(primitive_options)
        test_env.set_exploration_options(eigenoptions)

        # Initialise our agent and train it.
        agent = OptionAgent(
            env,
            test_env=test_env,
            epsilon=epsilon,
            macro_alpha=alpha,
            intra_option_alpha=alpha,
            gamma=gamma,
            n_step_updates=n_step_updates,
            default_action_value=default_action_value,
        )
        train_results, episode_test_results = agent.run_agent(
            num_epochs=num_epochs,
            epoch_length=epoch_length,
            test_interval=test_interval,
            test_length=test_episode_cutoff,
            test_runs=5,
            verbose_logging=False,
            epoch_eval=False,
            episodic_eval=True,
        )

        run_id = uuid.uuid1()

        # Save training performance.
        train_dir = f"./Training Results/Learning Curves/{env_name}/Train/{output_directory}/"
        Path(train_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{train_dir}/{experiment_id}-{run}-{run_id}.json", "w", encoding="utf-8") as f:
            json.dump(train_results, f, ensure_ascii=False, indent=4)

        # Save epoch-based evaluation performance.
        # epoch_test_dir = f"./Training Results/Learning Curves/{env_name}/Epoch/{output_directory}/"
        # Path(epoch_test_dir).mkdir(parents=True, exist_ok=True)
        # with open(f"{epoch_test_dir}/{experiment_id}-{run}-{run_id}.json", "w", encoding="utf-8") as f:
        #     json.dump(epoch_test_results, f, ensure_ascii=False, indent=4)

        # Save episode-based evaluation performance.
        episode_test_dir = f"./Training Results/Learning Curves/{env_name}/Episode/{output_directory}/"
        Path(episode_test_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{episode_test_dir}/{experiment_id}-{run}-{run_id}.json", "w", encoding="utf-8") as f:
            json.dump(episode_test_results, f, ensure_ascii=False, indent=4)

        gc.collect()


def train_agent_given_options(
    environment_args: Tuple[Type, Dict, str],
    epsilon: float,
    alpha: float,
    gamma: float,
    default_action_value: float,
    n_step_updates: bool,
    num_agents: int,
    test_interval: int,
    num_epochs: int,
    epoch_length: int,
    test_episode_cutoff: int,
    output_directory: str,
    experiment_id: int,
    options: List[BaseOption] = None,
    exploration_options: List[BaseOption] = None,
):
    """
    Given an arbitrary set of options, train a Macro-Q/Intra-Option Learning agent using these options.

    Args:
        environment_args (Tuple[Type, Dict, str]): The type, arguments, and name of the environment to train the agent in.
        epsilon (float): Exploration rate for the agent.
        alpha (float): Learning rate for the agent.
        gamma (float): Discount factor for the agent.
        default_action_value (float): The default value to assign to unseen state-action pairs.
        n_step_updates (bool): Whether or not to use n-step updates.
        num_agents (int): The number of agents to train.
        test_interval (int): The interval at which to test the agent, in epochs.
        num_epochs (int): The number of epochs to train the agent for.
        epoch_length (int): The length of each epoch, in primitive decision stages.
        test_episode_cutoff (int): The number of primitive decision stages after which a test episode is cut off.
        output_directory (str): The directory to save the training results in.
        experiment_id (int): The ID of the experiment being run.
        options (List[BaseOption], optional): The set of options available for the agent to choose. Defaults to None, in which only primitive options are made available.
        exploration_options (List[BaseOption], optional): The set of options available for the agent to explore using, but not explicitly choose. Defaults to None.
    """
    (EnvironmentType, kwargs, env_name) = environment_args

    # Create options representing primitive actions.
    env = EnvironmentType(**kwargs)
    env.reset()
    primitive_options = []
    for action in env.get_action_space():
        primitive_options.append(PrimitiveOption(action, env))

    if options is None:
        options = primitive_options
    else:
        options = primitive_options + options

    # Generate results.
    # Run Macro-Q Learning Agent.
    for run in tqdm(range(num_agents), desc="Options Agent"):
        # Initialise our environment.
        env = EnvironmentType(**kwargs)
        env.set_options(primitive_options + options)
        if exploration_options is not None:
            env.set_exploration_options(exploration_options)

        # Initialise our test environment.
        test_env = EnvironmentType(**kwargs)
        test_env.set_options(primitive_options + options)
        if exploration_options is not None:
            test_env.set_exploration_options(exploration_options)

        # Initialise our agent and train it.
        agent = OptionAgent(
            env,
            test_env=test_env,
            epsilon=epsilon,
            macro_alpha=alpha,
            intra_option_alpha=alpha,
            gamma=gamma,
            n_step_updates=n_step_updates,
            default_action_value=default_action_value,
        )
        train_results, episode_test_results = agent.run_agent(
            num_epochs=num_epochs,
            epoch_length=epoch_length,
            test_interval=test_interval,
            test_length=test_episode_cutoff,
            test_runs=5,
            verbose_logging=False,
            epoch_eval=False,
            episodic_eval=True,
        )

        run_id = uuid.uuid1()

        # Save training performance.
        train_dir = f"./Training Results/Learning Curves/{env_name}/Train/{output_directory}/"
        Path(train_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{train_dir}/{experiment_id}-{run}-{run_id}.json", "w", encoding="utf-8") as f:
            json.dump(train_results, f, ensure_ascii=False, indent=4)

        # Save epoch-based evaluation performance.
        # epoch_test_dir = f"./Training Results/Learning Curves/{env_name}/Epoch/{output_directory}/"
        # Path(epoch_test_dir).mkdir(parents=True, exist_ok=True)
        # with open(f"{epoch_test_dir}/{experiment_id}-{run}-{run_id}.json", "w", encoding="utf-8") as f:
        #     json.dump(epoch_test_results, f, ensure_ascii=False, indent=4)

        # Save episode-based evaluation performance.
        episode_test_dir = f"./Training Results/Learning Curves/{env_name}/Episode/{output_directory}/"
        Path(episode_test_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{episode_test_dir}/{experiment_id}-{run}-{run_id}.json", "w", encoding="utf-8") as f:
            json.dump(episode_test_results, f, ensure_ascii=False, indent=4)

        gc.collect()


def train_primitive_agent(
    environment_args: Tuple[Type, Dict, str],
    epsilon: float,
    alpha: float,
    gamma: float,
    default_action_value: float,
    num_agents: int,
    test_interval: int,
    num_epochs: int,
    epoch_length: int,
    test_episode_cutoff: int,
    output_directory: str,
    experiment_id: int,
):
    """
    Train a Macro-Q Learning agent using only primitive options.

    Args:
        environment_args (Tuple[Type, Dict, str]): The type, arguments, and name of the environment to train the agent in.
        epsilon (float): Exploration rate for the agent.
        alpha (float): Learning rate for the agent.
        gamma (float): Discount factor for the agent.
        default_action_value (float): The default value to assign to unseen state-action pairs.
        num_agents (int): The number of agents to train.
        test_interval (int): The interval at which to test the agent, in epochs.
        num_epochs (int): The number of epochs to train the agent for.
        epoch_length (int): The length of each epoch, in primitive decision stages.
        test_episode_cutoff (int): The number of primitive decision stages after which a test episode is cut off.
        output_directory (str): The directory to save the training results in.
        experiment_id (int): The ID of the experiment being run.
    """
    (EnvironmentType, kwargs, env_name) = environment_args

    # Initialise environment.
    env = EnvironmentType(**kwargs)
    env.reset()

    # Create options representing primitive actions.
    primitive_options = []
    for action in env.get_action_space():
        primitive_options.append(PrimitiveOption(action, env))
    env.set_options(primitive_options)

    # Train higher-level options.
    options = []
    options.extend(primitive_options)
    gc.collect()

    # Generate results.
    # Run Macro-Q Learning Agent
    for run in tqdm(range(num_agents), desc="Primitive Agent"):
        # Initialise our environment.
        env = EnvironmentType(**kwargs)
        env.set_options(options)

        test_env = EnvironmentType(**kwargs)
        test_env.set_options(options)

        # Initialise our agent and train it.
        agent = OptionAgent(
            env,
            test_env=test_env,
            epsilon=epsilon,
            macro_alpha=alpha,
            intra_option_alpha=alpha,
            gamma=gamma,
            default_action_value=default_action_value,
        )
        train_results, episode_test_results = agent.run_agent(
            num_epochs=num_epochs,
            epoch_length=epoch_length,
            test_interval=test_interval,
            test_length=test_episode_cutoff,
            test_runs=5,
            verbose_logging=False,
            epoch_eval=False,
            episodic_eval=True,
        )

        run_id = uuid.uuid1()

        # Save training performance.
        train_dir = f"./Training Results/Learning Curves/{env_name}/Train/{output_directory}/"
        Path(train_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{train_dir}/{experiment_id}-{run}-{run_id}.json", "w", encoding="utf-8") as f:
            json.dump(train_results, f, ensure_ascii=False, indent=4)

        # epoch_test_dir = f"./Training Results/Learning Curves/{env_name}/Epoch/{output_directory}/"
        # Path(epoch_test_dir).mkdir(parents=True, exist_ok=True)
        # with open(f"{epoch_test_dir}/{experiment_id}-{run}-{run_id}.json", "w", encoding="utf-8") as f:
        #     json.dump(epoch_test_results, f, ensure_ascii=False, indent=4)

        episode_test_dir = f"./Training Results/Learning Curves/{env_name}/Episode/{output_directory}/"
        Path(episode_test_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{episode_test_dir}/{experiment_id}-{run}-{run_id}.json", "w", encoding="utf-8") as f:
            json.dump(episode_test_results, f, ensure_ascii=False, indent=4)

        gc.collect()
