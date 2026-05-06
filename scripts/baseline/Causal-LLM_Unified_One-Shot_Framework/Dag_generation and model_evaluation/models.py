import numpy as np
from castle.algorithms import RL, GraNDAG, GES, PC, ICALiNGAM
import pandas as pd
from config_models import config

def initialize_models(input_dim, enabled_models=None):
    models = {}
    enabled = set(enabled_models) if enabled_models else set(config.keys())

    # Initialize causal_llm if present in config
    if "causal_llm" in config and "causal_llm" in enabled:
        # Lazy import keeps lighter model-only runs from requiring torch/transformers.
        from causal_llm import CausalDiscoveryLLM

        causal_llm_config = config["causal_llm"]
        output_dim = causal_llm_config["output_dim"](input_dim) 
        models["causal_llm"] = CausalDiscoveryLLM(
            input_dim=input_dim,
            output_dim=output_dim,
            model_path=causal_llm_config["model_path"]
        )

    # Initialize GraNDAG if present in config
    if "GraNDAG" in config and "GraNDAG" in enabled:
        gradag_config = config["GraNDAG"]
        iterations= gradag_config["iterations"]
        models["GraNDAG"] = GraNDAG(input_dim=input_dim,iterations=iterations)

    # Initialize RL if present in config
    if "RL" in config and "RL" in enabled:
        rl_config = config["RL"]
        models["RL"] = RL(nb_epoch=rl_config["nb_epoch"])

    # Initialize ICALiNGAM if present in config
    if "ICALiNGAM" in config and "ICALiNGAM" in enabled:
        icalingam_config = config["ICALiNGAM"]
        models["ICALiNGAM"] = ICALiNGAM(
            max_iter=icalingam_config["max_iter"],
            thresh=icalingam_config["thresh"]
        )

    # Initialize GES if present in config
    if "GES" in config and "GES" in enabled:
        models["GES"] = GES()

    # Initialize PC if present in config
    if "PC" in config and "PC" in enabled:
        models["PC"] = PC()

    return models

def preprocess_adj_matrix(adj_matrix, threshold=0.1):
    return np.where(np.abs(adj_matrix) > threshold, 1, 0)

def train_models(models, data, dataset_type, node_labels, ground_truth_name):
    results = {}
    for model_name, model in models.items():
        if model_name == 'causal_llm':
            model.learn(data)
            adj_matrix = model.causal_matrix(data)
        else:
            model.learn(data.astype(np.float64))
            adj_matrix = model.causal_matrix

        binary_adj_matrix = preprocess_adj_matrix(adj_matrix)
        results[model_name] = binary_adj_matrix

        adj_matrix_filename = f"{ground_truth_name}_{model_name}_{dataset_type}.csv" if dataset_type else f"{ground_truth_name}_{model_name}.csv"
        pd.DataFrame(binary_adj_matrix).to_csv(adj_matrix_filename, index=False, header=False)
    return results
