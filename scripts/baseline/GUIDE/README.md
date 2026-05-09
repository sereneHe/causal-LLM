# GUIDE : Generalized-Prior and Data Encoders for DAG Estimation

GUIDE combines generative priors, reinforcement learning, and a dual-encoder architecture to significantly improve scalability, reduce computational overhead, and effectively handle both mixed and nonlinear data types. At its core, GUIDE utilizes LLM-generated causal DAGs as initial priors, which provide valuable domain knowledge to guide the causal discovery process. This enables the framework to robustly support continuous and non-continuous data properties, addressing key limitations of traditional algorithms. By integrating these generative priors with reinforcement learning, GUIDE optimizes causal discovery in large-scale and complex datasets, ensuring accurate and efficient results across various real-world scenarios. Notably, the framework incorporates a constrained action space to strategically reduce the computational costs associated with exhaustive exploration, making the approach more efficient and applicable to diverse causal discovery tasks.


<img width="1180" height="768" alt="Causal DAG Discovery Workflow" src="https://github.com/user-attachments/assets/dfb5476f-2f72-4455-896b-e4a1ded80fdf" />

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Datasets](#datasets)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Overview

GUIDE is a novel framework for causal DAG discovery that synergizes generative knowledge with observational data. The framework addresses key challenges in causal discovery:

- **Scalability**: Handles large-scale datasets efficiently through constrained action spaces
- **Mixed Data Types**: Supports both continuous and discrete variables
- **Nonlinear Relationships**: Captures complex causal relationships through neural networks
- **Prior Knowledge Integration**: Leverages domain knowledge through generative priors
- **Computational Efficiency**: Reduces exploration costs through strategic constraints

## Key Features

### 🧠 Dual-Encoder Architecture
- **Data Encoder**: Processes input data features using transformer architecture
- **Adjacency Encoder**: Processes prior adjacency matrices for knowledge integration
- **Combined Processing**: Fuses data and structural information for optimal predictions

### 🎯 Reinforcement Learning
- **REINFORCE Algorithm**: Policy gradient method for learning DAG structures
- **Reward Function**: BIC-based scoring with cycle and structural penalties
- **Memory Management**: Efficient storage and retrieval of training experiences

### 📊 Comprehensive Evaluation
- **Multiple Metrics**: TPR, FDR, SHD, and other standard causal discovery metrics
- **Visualization**: Training progress plots and result analysis
- **Comparison Tools**: Easy comparison with ground truth structures

### 🔧 Flexible Configuration
- **Modular Design**: Easy to modify and extend components
- **Hyperparameter Tuning**: Comprehensive configuration options
- **Dataset Support**: Multiple built-in and custom dataset loaders

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for large datasets)

### Install Dependencies

```bash
pip install -r ../../../requirements.txt
```

### Verify Installation

```bash
python main.py --help
```

## Project Structure

```
GUIDE/
├── main.py                 # Main entry point
├── config.py              # Configuration parameters
├── models.py              # Neural network models
├── reward.py              # Reward calculation
├── trainer.py             # Training functions
├── data_loader.py         # Dataset loading
├── utils.py               # Utility functions
├── ../../../requirements.txt       # Consolidated dependencies
├── README.md              # This file
├── Datasets/              # Dataset directory
│   ├── PUBLIC/           # Public datasets
│   │   ├── Sachs/        # Sachs dataset
│   │   ├── Lucas/        # Lucas dataset
│   │   ├── Hepar2/       # Hepar2 dataset
|   |   ├── dream41/      # Dream41 dataset
|   |   ├── asia/         # Asia Dataset
|   |   └── alarm/        # Alarm Dataset
│   └── SYNTHETIC/        # Synthetic datasets
│       ├── gaussian_30nodes.csv
│       ├── gaussian_50nodes.csv
│       ├── non-gaussian_30nodes.csv
│       ├── non-gaussian_50nodes.csv
│       └── Ground truth/
└── outputs/               # Output directory (created automatically)
```

## Usage

### Quick Start

Run a quick test with synthetic data:

```bash
python main.py
```

### Basic Usage

Train on a specific dataset:

```bash
python main.py --dataset sachs --num_epochs 20 --actor_lr 0.001
```

### Advanced Usage

Use custom dataset and configuration:

```bash
python main.py \
    --data_path /path/to/data.csv \
    --adj_path /path/to/adjacency.csv \
    --hidden_dim 128 \
    --num_epochs 50 \
    --actor_lr 0.0005 \
    --prior_fraction 0.3 \
    --output_dir results/
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | 'sachs' | Dataset to use |
| `--data_path` | str | None | Custom data file path |
| `--adj_path` | str | None | Custom adjacency matrix path |
| `--hidden_dim` | int | 64 | Hidden dimension for model |
| `--num_epochs` | int | 10 | Number of training epochs |
| `--actor_lr` | float | 1e-3 | Learning rate for actor |
| `--prior_fraction` | float | 0.25 | Fraction of edges in partial prior |
| `--output_dir` | str | 'outputs/' | Output directory |
| `--save_model` | flag | False | Save trained model |
| `--verbose` | flag | False | Enable verbose output |

## Configuration

### Model Configuration

```python
MODEL_CONFIG = {
    'hidden_dim': 64,        # Hidden layer dimension
    'nheads': 8,             # Number of attention heads
    'dropout': 0.2,          # Dropout rate
    'num_layers': 3          # Number of transformer layers
}
```

### Training Configuration

```python
TRAINING_CONFIG = {
    'batch_size': 64,        # Batch size
    'gamma': 0.99,           # Discount factor
    'actor_lr': 1e-3,        # Actor learning rate
    'num_epochs': 10,        # Number of epochs
    'max_steps': 100,        # Maximum steps per episode
    'grad_clip_norm': 0.5    # Gradient clipping
}
```

### Reward Configuration

```python
REWARD_CONFIG = {
    'score_type': 'BIC_different_var',  # Score type
    'reg_type': 'LR',                   # Regression type
    'l1_graph_reg': 1.0,               # L1 regularization
    'lambda1': 1.0,                    # Cycle existence penalty
    'lambda2': 2.0,                    # Cycle magnitude penalty
    'lambda3': 0.5                     # Structural mismatch penalty
}
```


### Custom Datasets

To use your own dataset:

1. Prepare data as CSV file (samples × variables)
2. Prepare adjacency matrix as CSV or NPY file
3. Use `--data_path` and `--adj_path` arguments

## API Reference

### Core Classes

#### `DAGModel`
Main neural network model for DAG prediction.

```python
model = DAGModel(data_dim=10, hidden_dim=64, nheads=8)
```

#### `ReinforceAgent`
REINFORCE agent for learning DAG structures.

```python
agent = ReinforceAgent(model, actor_lr=1e-3, gamma=0.99, batch_size=64, partial_prior)
```

#### `get_Reward`
Reward calculator for evaluating DAG structures.

```python
reward_calc = get_Reward(batch_num=1, maxlen=10, dim=1, inputdata=data, ...)
```

### Utility Functions

#### `ensure_acyclic(adj_matrix)`
Remove cycles from adjacency matrix.

#### `create_partial_prior(actual_dag, fraction=0.25)`
Create partial prior knowledge from ground truth.

#### `count_accuracy(B_true, B_est)`
Calculate accuracy metrics for DAG comparison.

## Examples

### Example 1: Basic Training

```python
from models import DAGModel
from trainer import predict_dag_with_reinforce_no_threshold
from utils import create_partial_prior
import numpy as np

# Load data
data = np.random.randn(1000, 10)  # 1000 samples, 10 variables
true_dag = np.random.randint(0, 2, (10, 10))  # Random DAG

# Create priors
partial_prior1 = create_partial_prior(true_dag, fraction=0.25)
partial_prior = np.zeros_like(true_dag)

# Initialize model
model = DAGModel(data_dim=10, hidden_dim=64)

# Train and predict
final_dag, trained_model, best_adj, best_probs = predict_dag_with_reinforce_no_threshold(
    model=model,
    data=data,
    partial_prior=partial_prior,
    partial_prior1=partial_prior1,
    num_epochs=20,
    actor_lr=1e-3
)
```

### Example 2: Custom Dataset

```python
from data_loader import DataLoader

# Load custom dataset
loader = DataLoader("Datasets/")
data, true_dag = loader.load_custom_dataset("my_data.csv", "my_adjacency.csv")

# Preprocess data
data = loader.preprocess_data(data, normalize=True)

# Get dataset information
info = loader.get_dataset_info(data, true_dag)
print(f"Dataset info: {info}")
```

### Example 3: Evaluation

```python
from utils import count_accuracy

# Compare predicted and true DAGs
metrics = count_accuracy(true_dag, predicted_dag)

print(f"True Positive Rate: {metrics['tpr']:.4f}")
print(f"False Discovery Rate: {metrics['fdr']:.4f}")
print(f"Structural Hamming Distance: {metrics['shd']}")
```

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

