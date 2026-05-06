"""
Configuration file for GUIDE framework.
Contains hyperparameters and default settings.
"""

# Model Configuration
MODEL_CONFIG = {
    'hidden_dim': 64,
    'nheads': 8,
    'dropout': 0.2,
    'num_layers': 3
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 64,
    'gamma': 0.99,
    'actor_lr': 1e-3,
    'num_epochs': 10,
    'max_steps': 100,
    'grad_clip_norm': 0.5
}

# Reward Configuration
REWARD_CONFIG = {
    'score_type': 'BIC_different_var',
    'reg_type': 'LR',
    'l1_graph_reg': 1.0,
    'lambda1': 1.0,
    'lambda2': 2.0,
    'lambda3': 0.5,
    'sl': 0,
    'su': 1,
    'lambda1_upper': 1
}

# Data Configuration
DATA_CONFIG = {
    'threshold': 0.7,
    'prior_fraction': 0.25,
    'verbose': False
}

# Paths
PATHS = {
    'datasets_dir': 'Datasets/',
    'public_dir': 'Datasets/PUBLIC/',
    'synthetic_dir': 'Datasets/SYNTHETIC/',
    'output_dir': 'outputs/'
}
