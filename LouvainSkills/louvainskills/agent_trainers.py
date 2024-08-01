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

from simpleoptions import PrimitiveOption, OptionAgent


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
    environment_args,
    epsilon,
    alpha,
    gamma,
    default_action_value,
    n_step_updates,
    num_agents,
    num_epochs,
    epoch_length,
    test_episode_cutoff,
    option_training_num_rollouts,
    can_leave_initiation_set,
    output_directory,
    aggregate_graphs,
    stg,
    experiment_id,
):
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

        train_results, test_results = agent.run_agent(
            num_epochs=num_epochs,
            epoch_length=epoch_length,
            test_interval=1,
            test_length=test_episode_cutoff,
            test_runs=5,
            verbose_logging=False,
            episodic_eval=True,
        )

        # Write results to output file.
        test_dir = f"./Training Results/{env_name}/{output_directory}/"
        Path(test_dir).mkdir(parents=True, exist_ok=True)  # Testing performance.
        with open(f"{test_dir}/{experiment_id}-{run}-{uuid.uuid1()}.json", "w", encoding="utf-8") as f:
            json.dump(test_results, f, ensure_ascii=False, indent=4)

        train_dir = f"./Training Results/{env_name}/Train/{output_directory}/"
        Path(train_dir).mkdir(parents=True, exist_ok=True)  # Training performance.
        with open(f"{train_dir}/{experiment_id}-{run}-{uuid.uuid1()}.json", "w", encoding="utf-8") as f:
            json.dump(train_results, f, ensure_ascii=False, indent=4)

        gc.collect()


def train_single_level_agents(
    environment_args,
    epsilon,
    alpha,
    gamma,
    default_action_value,
    n_step_updates,
    num_agents,
    num_epochs,
    epoch_length,
    test_episode_cutoff,
    option_training_num_rollouts,
    can_leave_initiation_set,
    output_directory,
    aggregate_graphs,
    stg,
    experiment_id,
):
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
            train_results, test_results = agent.run_agent(
                num_epochs=num_epochs,
                epoch_length=epoch_length,
                test_interval=1,
                test_length=test_episode_cutoff,
                test_runs=5,
                verbose_logging=False,
                episodic_eval=True,
            )

            # Write results to output file.
            test_dir = f"./Training Results/{env_name}/{output_directory}/Level {level}/"
            Path(test_dir).mkdir(parents=True, exist_ok=True)  # Testing performance.
            with open(f"{test_dir}/{experiment_id}-{run}-{uuid.uuid1()}.json", "w", encoding="utf-8") as f:
                json.dump(test_results, f, ensure_ascii=False, indent=4)

            train_dir = f"./Training Results/{env_name}/Train/{output_directory}/Level {level}/"
            Path(train_dir).mkdir(parents=True, exist_ok=True)  # Training performance.
            with open(f"{train_dir}/{experiment_id}-{run}-{uuid.uuid1()}.json", "w", encoding="utf-8") as f:
                json.dump(train_results, f, ensure_ascii=False, indent=4)

            gc.collect()


def train_flat_agent(
    environment_args,
    epsilon,
    alpha,
    gamma,
    default_action_value,
    n_step_updates,
    num_agents,
    num_epochs,
    epoch_length,
    test_episode_cutoff,
    option_training_num_rollouts,
    can_leave_initiation_set,
    output_directory,
    aggregate_graphs,
    stg,
    experiment_id,
):
    (EnvironmentType, kwargs, env_name) = environment_args

    # Initialise environment.
    env = EnvironmentType(**kwargs)
    env.reset()

    # If there is more than one aggregate graph, remove the first one
    # because it corresponds to the singleton partition.
    if len(aggregate_graphs) > 1:
        aggregate_graphs = aggregate_graphs[1:]

    # Create a list of skills to train.
    skill_hierarchy = []
    for i, aggregate_graph in enumerate(aggregate_graphs):
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
        train_results, test_results = agent.run_agent(
            num_epochs=num_epochs,
            epoch_length=epoch_length,
            test_interval=1,
            test_length=test_episode_cutoff,
            test_runs=5,
            verbose_logging=False,
            episodic_eval=True,
        )

        # Write results to output file.
        test_dir = f"./Training Results/{env_name}/{output_directory}/"
        Path(test_dir).mkdir(parents=True, exist_ok=True)  # Testing performance.
        with open(f"{test_dir}/{experiment_id}-{run}-{uuid.uuid1()}.json", "w", encoding="utf-8") as f:
            json.dump(test_results, f, ensure_ascii=False, indent=4)

        train_dir = f"./Training Results/{env_name}/Train/{output_directory}/"
        Path(train_dir).mkdir(parents=True, exist_ok=True)  # Training performance.
        with open(f"{train_dir}/{experiment_id}-{run}-{uuid.uuid1()}.json", "w", encoding="utf-8") as f:
            json.dump(train_results, f, ensure_ascii=False, indent=4)

        gc.collect()


def train_betweenness_agent(
    environment_args,
    epsilon,
    alpha,
    gamma,
    default_action_value,
    n_step_updates,
    num_agents,
    num_epochs,
    epoch_length,
    test_episode_cutoff,
    option_training_num_rollouts,
    output_directory,
    subgoals,
    centralities,
    n_options,
    initiation_set_size,
    stg,
    experiment_id,
):
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
        train_results, test_results = agent.run_agent(
            num_epochs=num_epochs,
            epoch_length=epoch_length,
            test_interval=1,
            test_length=test_episode_cutoff,
            test_runs=5,
            verbose_logging=False,
            episodic_eval=True,
        )

        # Write results to output file.
        test_dir = f"./Training Results/{env_name}/{output_directory}/"
        Path(test_dir).mkdir(parents=True, exist_ok=True)  # Testing performance.
        with open(f"{test_dir}/{experiment_id}-{run}-{uuid.uuid1()}.json", "w", encoding="utf-8") as f:
            json.dump(test_results, f, ensure_ascii=False, indent=4)

        train_dir = f"./Training Results/{env_name}/Train/{output_directory}/"
        Path(train_dir).mkdir(parents=True, exist_ok=True)  # Training performance.
        with open(f"{train_dir}/{experiment_id}-{run}-{uuid.uuid1()}.json", "w", encoding="utf-8") as f:
            json.dump(train_results, f, ensure_ascii=False, indent=4)

        gc.collect()


def train_eigenoptions_agent(
    environment_args,
    epsilon,
    alpha,
    gamma,
    default_action_value,
    n_step_updates,
    num_agents,
    num_epochs,
    epoch_length,
    test_episode_cutoff,
    output_directory,
    pvfs,
    stg,
    experiment_id,
    training_env_args=None,
):
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
        train_results, test_results = agent.run_agent(
            num_epochs=num_epochs,
            epoch_length=epoch_length,
            test_interval=1,
            test_length=test_episode_cutoff,
            test_runs=5,
            verbose_logging=False,
            episodic_eval=True,
        )

        # Write results to output file.
        test_dir = f"./Training Results/{env_name}/{output_directory}/"
        Path(test_dir).mkdir(parents=True, exist_ok=True)  # Testing performance.
        with open(f"{test_dir}/{experiment_id}-{run}-{uuid.uuid1()}.json", "w", encoding="utf-8") as f:
            json.dump(test_results, f, ensure_ascii=False, indent=4)

        train_dir = f"./Training Results/{env_name}/Train/{output_directory}/"
        Path(train_dir).mkdir(parents=True, exist_ok=True)  # Training performance.
        with open(f"{train_dir}/{experiment_id}-{run}-{uuid.uuid1()}.json", "w", encoding="utf-8") as f:
            json.dump(train_results, f, ensure_ascii=False, indent=4)

        gc.collect()


def train_agent_given_options(
    environment_args,
    epsilon,
    alpha,
    gamma,
    default_action_value,
    n_step_updates,
    num_agents,
    num_epochs,
    epoch_length,
    test_episode_cutoff,
    output_directory,
    experiment_id,
    options=None,
    exploration_options=None,
):
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
        train_results, test_results = agent.run_agent(
            num_epochs=num_epochs,
            epoch_length=epoch_length,
            test_interval=1,
            test_length=test_episode_cutoff,
            test_runs=5,
            verbose_logging=False,
            episodic_eval=True,
        )

        # Write results to output file.
        test_dir = f"./Training Results/{env_name}/{output_directory}/"
        Path(test_dir).mkdir(parents=True, exist_ok=True)  # Testing performance.
        with open(f"{test_dir}/{experiment_id}-{run}-{uuid.uuid1()}.json", "w", encoding="utf-8") as f:
            json.dump(test_results, f, ensure_ascii=False, indent=4)

        train_dir = f"./Training Results/{env_name}/Train/{output_directory}/"
        Path(train_dir).mkdir(parents=True, exist_ok=True)  # Training performance.
        with open(f"{train_dir}/{experiment_id}-{run}-{uuid.uuid1()}.json", "w", encoding="utf-8") as f:
            json.dump(train_results, f, ensure_ascii=False, indent=4)

        gc.collect()


def train_primitive_agent(
    environment_args,
    epsilon,
    alpha,
    gamma,
    default_action_value,
    num_agents,
    num_epochs,
    epoch_length,
    test_episode_cutoff,
    output_directory,
    experiment_id,
):
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
        train_results, test_results = agent.run_agent(
            num_epochs=num_epochs,
            epoch_length=epoch_length,
            test_interval=1,
            test_length=test_episode_cutoff,
            test_runs=5,
            verbose_logging=False,
            episodic_eval=True,
        )

        # Write results to output file.
        test_dir = f"./Training Results/{env_name}/{output_directory}/"
        Path(test_dir).mkdir(parents=True, exist_ok=True)  # Testing performance.
        with open(f"{test_dir}/{experiment_id}-{run}-{uuid.uuid1()}.json", "w", encoding="utf-8") as f:
            json.dump(test_results, f, ensure_ascii=False, indent=4)

        train_dir = f"./Training Results/{env_name}/Train/{output_directory}/"
        Path(train_dir).mkdir(parents=True, exist_ok=True)  # Training performance.
        with open(f"{train_dir}/{experiment_id}-{run}-{uuid.uuid1()}.json", "w", encoding="utf-8") as f:
            json.dump(train_results, f, ensure_ascii=False, indent=4)

        gc.collect()
