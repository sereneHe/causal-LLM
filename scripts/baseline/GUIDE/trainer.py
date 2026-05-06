"""
Training module for GUIDE framework.
Contains training functions and main prediction pipeline.
"""

import torch
import numpy as np
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
from scipy.linalg import expm as matrix_exponential

from models import DAGModel, ReinforceAgent
from reward import get_Reward
from utils import ensure_acyclic, graph_prunned_by_coef, plot_training_progress
from config import TRAINING_CONFIG, REWARD_CONFIG, MODEL_CONFIG


def train_dag_model_with_reinforce(model, data, partial_prior, partial_prior1, num_epochs, actor_lr):
    """
    Train DAG model using REINFORCE algorithm.
    
    Args:
        model: DAGModel instance
        data: input data matrix
        partial_prior: generative prior matrix
        partial_prior1: partial prior knowledge matrix
        num_epochs: number of training epochs
        actor_lr: learning rate for actor
        
    Returns:
        trained_model: trained model
        best_adj_matrix: best adjacency matrix found
        best_adj_probs: best adjacency probabilities
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Hyperparams
    batch_size = TRAINING_CONFIG['batch_size']
    gamma = TRAINING_CONFIG['gamma']

    # Agent
    agent = ReinforceAgent(
        model=model,
        actor_lr=actor_lr,
        gamma=gamma,
        batch_size=batch_size,
        partial_prior=torch.tensor(partial_prior, dtype=torch.float32, device=device)
    )

    # Reward calculator
    reward_calculator = get_Reward(
        batch_num=1,
        maxlen=model.data_dim,
        dim=1,
        inputdata=data,
        sl=REWARD_CONFIG['sl'],
        su=REWARD_CONFIG['su'],
        lambda1_upper=REWARD_CONFIG['lambda1_upper'],
        score_type=REWARD_CONFIG['score_type'],
        reg_type=REWARD_CONFIG['reg_type'],
        l1_graph_reg=REWARD_CONFIG['l1_graph_reg'],
        verbose_flag=False
    )

    best_reward = -float("inf")
    best_adj_matrix = None

    all_rewards = []
    actor_loss_history = []
    edges_over_time = []
    cycles_over_time = []

    env_states = deque(data)

    for epoch in tqdm(range(num_epochs)):
        max_steps = TRAINING_CONFIG['max_steps']
        step_count = 0
        done = False
        state = env_states[0]
        episode_reward = 0.0

        while not done:
            # Choose action
            adj_matrix, log_prob = agent.choose_action(state)

            # Threshold adjacency
            adj_probs = torch.sigmoid(adj_matrix)
            mask_eye = 1.0 - torch.eye(model.data_dim, device=adj_probs.device)
            adj_probs = adj_probs * mask_eye
            adj_matrix = (adj_probs >= 0.7).float()  # threshold
            
            # Ensure acyclicity
            adj_matrix = ensure_acyclic(adj_matrix)

            # Calculate reward
            reward, score, cyc, penalty = reward_calculator.calculate_reward_single_graph(
                adj_matrix, partial_prior1, 
                lambda1=REWARD_CONFIG['lambda1'], 
                lambda2=REWARD_CONFIG['lambda2'], 
                lambda3=REWARD_CONFIG['lambda3']
            )

            if reward > best_reward:
                best_reward = reward
                best_adj_matrix = adj_matrix.detach().cpu().numpy().copy()
                best_adj_probs = adj_probs.detach().cpu().numpy().copy()

            episode_reward += reward

            # Store in memory
            agent.remember(
                torch.tensor(state, dtype=torch.float32).to(device),
                adj_matrix.to(device),
                log_prob.to(device),
                reward,
                done
            )

            step_count += 1
            if step_count >= max_steps or step_count >= len(data):
                done = True
            else:
                next_idx = step_count % len(data)
                state = env_states[next_idx]

        all_rewards.append(episode_reward)

        # Update model
        loss_val = agent.learn()
        if loss_val is not None:
            actor_loss_history.append(loss_val)

        # Log metrics
        if best_adj_matrix is not None:
            last_adj_t = torch.tensor(best_adj_matrix, dtype=torch.float32, device=device)
            num_edges = last_adj_t.sum().item()
            cyc_val = np.trace(matrix_exponential(last_adj_t.cpu().numpy())) - model.data_dim
            edges_over_time.append(num_edges)
            cycles_over_time.append(cyc_val)

        if epoch % 2 == 0:
            print(f"Epoch {epoch}, EpisodeReward={episode_reward:.4f}, BestReward={best_reward:.4f}")

    print("=== Finished Training ===")
    print(f"Best adjacency reward found: {best_reward:.4f}")

    # Plot training progress
    plot_training_progress(all_rewards, actor_loss_history, edges_over_time, cycles_over_time)

    return model, best_adj_matrix, best_adj_probs


def predict_dag_with_reinforce_no_threshold(
    model,
    data,
    partial_prior,
    partial_prior1,
    num_epochs=10,
    actor_lr=1e-3
):
    """
    Complete pipeline for DAG prediction using REINFORCE.
    
    Args:
        model: DAGModel instance
        data: input data matrix
        partial_prior: generative prior matrix
        partial_prior1: partial prior knowledge matrix
        num_epochs: number of training epochs
        actor_lr: learning rate for actor
        
    Returns:
        final_dag_np: final predicted DAG
        trained_model: trained model
        best_adj_matrix: best adjacency matrix from training
        best_adj_probs: best adjacency probabilities
    """
    # Train the model
    trained_model, best_adj_matrix, best_adj_probs = train_dag_model_with_reinforce(
        model=model,
        data=data,
        partial_prior=partial_prior,
        partial_prior1=partial_prior1,
        num_epochs=num_epochs,
        actor_lr=actor_lr
    )

    # Prune edges based on regression coefficients
    pruned_np = graph_prunned_by_coef(best_adj_matrix, data)

    # Ensure final adjacency is acyclic
    pruned_pt = ensure_acyclic(torch.tensor(pruned_np, dtype=torch.float32))
    final_dag_np = pruned_pt.cpu().numpy()

    print("Final DAG shape:", final_dag_np.shape)
    print("Sum of edges in the final DAG:", final_dag_np.sum())
    
    return final_dag_np, trained_model, best_adj_matrix, best_adj_probs


def evaluate_model_performance(predicted_dag, true_dag):
    """
    Evaluate model performance using various metrics.
    
    Args:
        predicted_dag: predicted adjacency matrix
        true_dag: true adjacency matrix
        
    Returns:
        metrics: dictionary of evaluation metrics
    """
    from utils import count_accuracy
    
    metrics = count_accuracy(true_dag, predicted_dag)
    
    print("=== Evaluation Results ===")
    print(f"True Positive Rate (TPR): {metrics['tpr']:.4f}")
    print(f"False Discovery Rate (FDR): {metrics['fdr']:.4f}")
    print(f"Structural Hamming Distance (SHD): {metrics['shd']}")
    print(f"True Positives: {metrics['tp']}")
    print(f"False Positives: {metrics['fp']}")
    print(f"True Negatives: {metrics['tn']}")
    print(f"False Negatives: {metrics['fn']}")
    
    return metrics


def run_experiment(dataset_name, datasets_dir="Datasets/", num_epochs=10, actor_lr=1e-3):
    """
    Run a complete experiment on a dataset.
    
    Args:
        dataset_name: name of the dataset to use
        datasets_dir: path to datasets directory
        num_epochs: number of training epochs
        actor_lr: learning rate for actor
        
    Returns:
        results: dictionary containing experiment results
    """
    from data_loader import load_dataset_by_name
    from utils import create_partial_prior, load_generative_prior
    
    print(f"Running experiment on {dataset_name} dataset...")
    
    # Load dataset
    data, true_dag = load_dataset_by_name(dataset_name, datasets_dir)
    if data is None or true_dag is None:
        print(f"Failed to load dataset: {dataset_name}")
        return None
    
    print(f"Dataset loaded: {data.shape[0]} samples, {data.shape[1]} variables")
    
    # Create priors
    partial_prior1 = create_partial_prior(true_dag, fraction=0.25)
    
    # Load generative prior (if available)
    generative_prior_path = f"generative_prior_{dataset_name}.npy"
    partial_prior = load_generative_prior(generative_prior_path)
    if partial_prior is None:
        partial_prior = np.zeros_like(true_dag)
    
    # Initialize model
    model = DAGModel(data_dim=data.shape[1], hidden_dim=MODEL_CONFIG['hidden_dim'])
    
    # Run prediction
    final_dag, trained_model, best_adj, best_probs = predict_dag_with_reinforce_no_threshold(
        model=model,
        data=data,
        partial_prior=partial_prior,
        partial_prior1=partial_prior1,
        num_epochs=num_epochs,
        actor_lr=actor_lr
    )
    
    # Evaluate performance
    metrics = evaluate_model_performance(final_dag, true_dag)
    
    results = {
        'dataset_name': dataset_name,
        'final_dag': final_dag,
        'true_dag': true_dag,
        'metrics': metrics,
        'model': trained_model,
        'best_adj': best_adj,
        'best_probs': best_probs
    }
    
    return results
