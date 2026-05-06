"""
Main entry point for GUIDE framework.
"""

import argparse
import os
import sys
from typing import Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from models import DAGModel
from data_loader import load_dataset_by_name, DataLoader
from trainer import run_experiment, predict_dag_with_reinforce_no_threshold
from utils import create_partial_prior, load_generative_prior, save_results
from config import MODEL_CONFIG, TRAINING_CONFIG, REWARD_CONFIG


def main():
    """Main function for GUIDE framework."""
    parser = argparse.ArgumentParser(description='GUIDE: Generalized-Prior and Data Encoders for DAG Estimation')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='sachs', 
                       choices=['sachs', 'lucas', 'hepar2', 'gaussian_30', 'gaussian_50', 'non_gaussian_30', 'non_gaussian_50'],
                       help='Dataset to use for training')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Custom data file path (CSV)')
    parser.add_argument('--adj_path', type=str, default=None,
                       help='Custom adjacency matrix file path (CSV or NPY)')
    parser.add_argument('--datasets_dir', type=str, default='Datasets/',
                       help='Path to datasets directory')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension for the model')
    parser.add_argument('--nheads', type=int, default=8,
                       help='Number of attention heads')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--actor_lr', type=float, default=1e-3,
                       help='Learning rate for actor')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    
    # Prior arguments
    parser.add_argument('--prior_fraction', type=float, default=0.25,
                       help='Fraction of edges to keep as known in partial prior')
    parser.add_argument('--generative_prior_path', type=str, default=None,
                       help='Path to generative prior file')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs/',
                       help='Output directory for results')
    parser.add_argument('--save_model', action='store_true',
                       help='Save the trained model')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("GUIDE: Generalized-Prior and Data Encoders for DAG Estimation")
    print("=" * 60)
    
    # Load dataset
    if args.data_path and args.adj_path:
        # Custom dataset
        loader = DataLoader(args.datasets_dir)
        data, true_dag = loader.load_custom_dataset(args.data_path, args.adj_path)
        dataset_name = "custom"
    else:
        # Predefined dataset
        data, true_dag = load_dataset_by_name(args.dataset, args.datasets_dir)
        dataset_name = args.dataset
    
    if data is None or true_dag is None:
        print(f"Error: Failed to load dataset '{dataset_name}'")
        return
    
    print(f"Dataset: {dataset_name}")
    print(f"Data shape: {data.shape}")
    print(f"True DAG shape: {true_dag.shape}")
    print(f"Number of true edges: {int(np.sum(true_dag > 0.5))}")
    
    # Create priors
    print("\nCreating priors...")
    partial_prior1 = create_partial_prior(true_dag, fraction=args.prior_fraction)
    
    # Load generative prior
    if args.generative_prior_path and os.path.exists(args.generative_prior_path):
        partial_prior = load_generative_prior(args.generative_prior_path)
        print(f"Loaded generative prior from {args.generative_prior_path}")
    else:
        partial_prior = np.zeros_like(true_dag)
        print("Using zero matrix as generative prior")
    
    # Initialize model
    print(f"\nInitializing model with hidden_dim={args.hidden_dim}...")
    model = DAGModel(
        data_dim=data.shape[1], 
        hidden_dim=args.hidden_dim,
        nheads=args.nheads
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Run training and prediction
    print(f"\nStarting training for {args.num_epochs} epochs...")
    final_dag, trained_model, best_adj, best_probs = predict_dag_with_reinforce_no_threshold(
        model=model,
        data=data,
        partial_prior=partial_prior,
        partial_prior1=partial_prior1,
        num_epochs=args.num_epochs,
        actor_lr=args.actor_lr
    )
    
    # Evaluate results
    print("\nEvaluating results...")
    from utils import count_accuracy
    metrics = count_accuracy(true_dag, final_dag)
    
    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"True Positive Rate (TPR): {metrics['tpr']:.4f}")
    print(f"False Discovery Rate (FDR): {metrics['fdr']:.4f}")
    print(f"Structural Hamming Distance (SHD): {metrics['shd']}")
    print(f"True Positives: {metrics['tp']}")
    print(f"False Positives: {metrics['fp']}")
    print(f"True Negatives: {metrics['tn']}")
    print(f"False Negatives: {metrics['fn']}")
    print(f"Predicted edges: {int(np.sum(final_dag > 0.5))}")
    print(f"True edges: {int(np.sum(true_dag > 0.5))}")
    
    # Save results
    results = {
        'dataset_name': dataset_name,
        'final_dag': final_dag,
        'true_dag': true_dag,
        'metrics': metrics,
        'best_adj': best_adj,
        'best_probs': best_probs,
        'config': {
            'hidden_dim': args.hidden_dim,
            'num_epochs': args.num_epochs,
            'actor_lr': args.actor_lr,
            'prior_fraction': args.prior_fraction
        }
    }
    
    results_path = os.path.join(args.output_dir, f"results_{dataset_name}.npy")
    save_results(results, results_path)
    
    if args.save_model:
        model_path = os.path.join(args.output_dir, f"model_{dataset_name}.pth")
        torch.save(trained_model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    print(f"\nResults saved to {results_path}")
    print("Experiment completed successfully!")



if __name__ == "__main__":
    main()
