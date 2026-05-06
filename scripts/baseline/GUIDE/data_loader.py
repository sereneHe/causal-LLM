"""
Data loading and preprocessing module for GUIDE framework.
Handles loading of various datasets including synthetic and real-world data.
"""

import numpy as np
import pandas as pd
import networkx as nx
from cdt.data import load_dataset
import os
from typing import Tuple, Optional


class DataLoader:
    """
    Data loader class for GUIDE framework.
    Supports loading of synthetic datasets, real-world datasets, and custom data.
    """
    
    def __init__(self, datasets_dir: str = "Datasets/"):
        """
        Initialize data loader.
        
        Args:
            datasets_dir: path to datasets directory
        """
        self.datasets_dir = datasets_dir
        self.public_dir = os.path.join(datasets_dir, "PUBLIC")
        self.synthetic_dir = os.path.join(datasets_dir, "SYNTHETIC")
    
    def load_sachs_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load Sachs dataset (11 nodes).
        
        Returns:
            data: data matrix
            adj_matrix: ground truth adjacency matrix
        """
        try:
            s_data, s_graph = load_dataset('dream4-1')
            actual_dag = nx.to_numpy_array(s_graph, nodelist=s_data.columns)
            loaded_data = s_data.to_numpy()
            return loaded_data, actual_dag
        except Exception as e:
            print(f"Error loading Sachs dataset: {e}")
            return None, None
    
    def load_synthetic_gaussian(self, nodes: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load synthetic Gaussian dataset.
        
        Args:
            nodes: number of nodes (30 or 50)
            
        Returns:
            data: data matrix
            adj_matrix: ground truth adjacency matrix
        """
        try:
            data_file = os.path.join(self.synthetic_dir, f"gaussian_{nodes}nodes.csv")
            adj_file = os.path.join(self.synthetic_dir, "Ground truth", f"adj_matrix_{nodes}nodes_linear.csv")
            
            if not os.path.exists(data_file) or not os.path.exists(adj_file):
                print(f"Files not found for {nodes} nodes dataset")
                return None, None
            
            loaded_data = pd.read_csv(data_file, header=0)
            adj = pd.read_csv(adj_file, header=0)
            
            data = loaded_data.to_numpy()
            actual_dag = adj.to_numpy()
            
            return data, actual_dag
        except Exception as e:
            print(f"Error loading synthetic Gaussian dataset: {e}")
            return None, None
    
    def load_synthetic_non_gaussian(self, nodes: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load synthetic non-Gaussian dataset.
        
        Args:
            nodes: number of nodes (30 or 50)
            
        Returns:
            data: data matrix
            adj_matrix: ground truth adjacency matrix
        """
        try:
            data_file = os.path.join(self.synthetic_dir, f"non-gaussian_{nodes}nodes.csv")
            adj_file = os.path.join(self.synthetic_dir, "Ground truth", f"adj_matrix_{nodes}nodes_linear.csv")
            
            if not os.path.exists(data_file) or not os.path.exists(adj_file):
                print(f"Files not found for {nodes} nodes dataset")
                return None, None
            
            loaded_data = pd.read_csv(data_file, header=0)
            adj = pd.read_csv(adj_file, header=0)
            
            data = loaded_data.to_numpy()
            actual_dag = adj.to_numpy()
            
            return data, actual_dag
        except Exception as e:
            print(f"Error loading synthetic non-Gaussian dataset: {e}")
            return None, None
    
    def load_lucas_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load Lucas dataset.
        
        Returns:
            data: data matrix
            adj_matrix: ground truth adjacency matrix
        """
        try:
            data_file = os.path.join(self.public_dir, "Lucas", "lucas.csv")
            adj_file = os.path.join(self.public_dir, "Lucas", "adj_matrix_lucas.npy")
            
            if not os.path.exists(data_file) or not os.path.exists(adj_file):
                print("Lucas dataset files not found")
                return None, None
            
            loaded_data = pd.read_csv(data_file, header=0)
            actual_dag = np.load(adj_file)
            
            data = loaded_data.to_numpy()
            
            return data, actual_dag
        except Exception as e:
            print(f"Error loading Lucas dataset: {e}")
            return None, None
    
    def load_hepar2_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load Hepar2 dataset.
        
        Returns:
            data: data matrix
            adj_matrix: ground truth adjacency matrix
        """
        try:
            data_file = os.path.join(self.public_dir, "Hepar2", "hepar2.csv")
            adj_file = os.path.join(self.public_dir, "Hepar2", "hepar2adj.npy")
            
            if not os.path.exists(data_file) or not os.path.exists(adj_file):
                print("Hepar2 dataset files not found")
                return None, None
            
            loaded_data = pd.read_csv(data_file, header=0)
            actual_dag = np.load(adj_file)
            
            data = loaded_data.to_numpy()
            
            return data, actual_dag
        except Exception as e:
            print(f"Error loading Hepar2 dataset: {e}")
            return None, None
    
    def load_custom_dataset(self, data_path: str, adj_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load custom dataset from files.
        
        Args:
            data_path: path to data file (CSV)
            adj_path: path to adjacency matrix file (CSV or NPY)
            
        Returns:
            data: data matrix
            adj_matrix: ground truth adjacency matrix
        """
        try:
            # Load data
            if data_path.endswith('.csv'):
                loaded_data = pd.read_csv(data_path, header=0)
                data = loaded_data.to_numpy()
            else:
                data = np.load(data_path)
            
            # Load adjacency matrix
            if adj_path.endswith('.csv'):
                adj = pd.read_csv(adj_path, header=0)
                actual_dag = adj.to_numpy()
            else:
                actual_dag = np.load(adj_path)
            
            return data, actual_dag
        except Exception as e:
            print(f"Error loading custom dataset: {e}")
            return None, None
    
    def preprocess_data(self, data: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Preprocess data (normalization, etc.).
        
        Args:
            data: input data matrix
            normalize: whether to normalize the data
            
        Returns:
            processed_data: preprocessed data matrix
        """
        if normalize:
            # Z-score normalization
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            data = (data - mean) / (std + 1e-8)
        
        return data
    
    def get_dataset_info(self, data: np.ndarray, adj_matrix: np.ndarray) -> dict:
        """
        Get information about the dataset.
        
        Args:
            data: data matrix
            adj_matrix: adjacency matrix
            
        Returns:
            info: dictionary containing dataset information
        """
        info = {
            'n_samples': data.shape[0],
            'n_variables': data.shape[1],
            'n_edges': int(np.sum(adj_matrix > 0.5)),
            'sparsity': 1 - (np.sum(adj_matrix > 0.5) / (adj_matrix.shape[0] * (adj_matrix.shape[0] - 1))),
            'data_range': (np.min(data), np.max(data)),
            'data_mean': np.mean(data),
            'data_std': np.std(data)
        }
        return info


def load_dataset_by_name(dataset_name: str, datasets_dir: str = "Datasets/") -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to load dataset by name.
    
    Args:
        dataset_name: name of the dataset to load
        datasets_dir: path to datasets directory
        
    Returns:
        data: data matrix
        adj_matrix: ground truth adjacency matrix
    """
    loader = DataLoader(datasets_dir)
    
    if dataset_name.lower() == 'sachs':
        return loader.load_sachs_dataset()
    elif dataset_name.lower() == 'lucas':
        return loader.load_lucas_dataset()
    elif dataset_name.lower() == 'hepar2':
        return loader.load_hepar2_dataset()
    elif dataset_name.lower() == 'gaussian_30':
        return loader.load_synthetic_gaussian(30)
    elif dataset_name.lower() == 'gaussian_50':
        return loader.load_synthetic_gaussian(50)
    elif dataset_name.lower() == 'non_gaussian_30':
        return loader.load_synthetic_non_gaussian(30)
    elif dataset_name.lower() == 'non_gaussian_50':
        return loader.load_synthetic_non_gaussian(50)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
