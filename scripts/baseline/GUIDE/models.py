"""
Neural network models for GUIDE framework.
Contains DAGModel, ReinforceMemory, and ReinforceAgent classes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class DAGModel(nn.Module):
    """
    Dual-encoder DAG model that processes both data and adjacency matrix.
    
    The model uses two transformer encoders:
    1. Data encoder: processes input data features
    2. Adjacency encoder: processes prior adjacency matrix
    """
    
    def __init__(self, data_dim, hidden_dim, nheads=8):
        super(DAGModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_dim = data_dim
        self.adj_dim = data_dim * data_dim
        self.hidden_dim = hidden_dim

        # Linear layer to project input data to `hidden_dim` for transformer
        self.data_projection = nn.Linear(data_dim, hidden_dim).to(self.device)

        # Transformer Encoder for processing data
        data_encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nheads, dropout=0.2
        )
        self.data_encoder = TransformerEncoder(data_encoder_layer, num_layers=3).to(self.device)

        # Transformer Encoder for processing adjacency matrix
        adj_encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nheads, dropout=0.2
        )
        self.adj_encoder = TransformerEncoder(adj_encoder_layer, num_layers=3).to(self.device)

        # MLPs to process adjacency features and combine
        self.adj_processor = nn.Sequential(
            nn.Linear(hidden_dim * data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ).to(self.device)

        self.combined_processor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.adj_dim),
        ).to(self.device)

    def forward(self, data, prior_adj):
        """
        Forward pass through the dual-encoder architecture.
        
        Args:
            data: [batch_size, data_dim] - input data
            prior_adj: [batch_size, data_dim, data_dim] or [data_dim, data_dim] - prior adjacency
            
        Returns:
            adj_output: [batch_size, data_dim, data_dim] - predicted adjacency matrix
        """
        data = data.to(self.device)

        # 1) Project data to `hidden_dim`
        data_projected = self.data_projection(data)  # [B, hidden_dim]

        # Expand to sequence format for transformer
        batch_size = data.size(0)
        data_projected = data_projected.unsqueeze(1).expand(-1, self.data_dim, -1)  # [B, d, hidden_dim]
        x_data = self.data_encoder(data_projected)  # [B, d, hidden_dim]
        x_data = x_data.mean(dim=1)  # Aggregate along the sequence dimension [B, hidden_dim]

        # 2) Transformer Encoder on prior adjacency
        if prior_adj.dim() == 2:
            prior_adj = prior_adj.unsqueeze(0).expand(batch_size, -1, -1)
        prior_adj = prior_adj.to(self.device)

        # Project adjacency matrix to match `hidden_dim`
        prior_adj_projected = self.data_projection(prior_adj)  # Project to [B, d, hidden_dim]

        # Pass through adjacency transformer encoder
        x_adj = self.adj_encoder(prior_adj_projected)  # [B, d, hidden_dim]
        x_adj_flat = x_adj.view(batch_size, -1)  # [B, d*hidden_dim]
        x_adj_processed = self.adj_processor(x_adj_flat)  # [B, hidden_dim]

        # Combine data features + adjacency features
        x_combined = torch.cat([x_data, x_adj_processed], dim=1)  # [B, hidden_dim*2]

        # Predict adjacency logits
        adj_output = self.combined_processor(x_combined)   # [B, d*d]
        adj_output = adj_output.view(batch_size, self.data_dim, self.data_dim)
        return adj_output


class ReinforceMemory:
    """
    Memory buffer for REINFORCE algorithm.
    Stores states, adjacency matrices, log probabilities, rewards, and done flags.
    """
    
    def __init__(self):
        self.states = []
        self.adj_matrices = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def store_memory(self, state, adj_matrix, log_prob, reward, done):
        """Store a transition in memory."""
        self.states.append(state)
        self.adj_matrices.append(adj_matrix)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        """Clear all stored memories."""
        self.states = []
        self.adj_matrices = []
        self.log_probs = []
        self.rewards = []
        self.dones = []


class ReinforceAgent:
    """
    REINFORCE agent for learning DAG structure through reinforcement learning.
    """
    
    def __init__(self, model, actor_lr, gamma, batch_size, partial_prior):
        self.model = model
        self.gamma = gamma
        self.batch_size = batch_size
        self.partial_prior = partial_prior

        self.actor_optimizer = optim.AdamW(self.model.parameters(), lr=actor_lr)
        self.memory = ReinforceMemory()
        self.device = self.model.device

    def remember(self, state, adj_matrix, log_prob, reward, done):
        """Store a transition in memory."""
        self.memory.store_memory(state, adj_matrix, log_prob, reward, done)

    def choose_action(self, state):
        """
        Choose an action (adjacency matrix) based on current state.
        
        Args:
            state: current data state
            
        Returns:
            adj_sampled: sampled adjacency matrix
            log_prob: log probability of the sampled action
        """
        device = self.model.device
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state_tensor = state.unsqueeze(0).to(device)

        # Forward pass with partial_prior
        adj_output = self.model(state_tensor, self.partial_prior)
        adj_probs = torch.sigmoid(adj_output)
        mask = 1.0 - torch.eye(self.model.data_dim, device=device)
        adj_probs = adj_probs * mask

        dist = torch.distributions.Bernoulli(probs=adj_probs)
        adj_sampled = dist.sample()
        log_prob = dist.log_prob(adj_sampled).sum()
        return adj_sampled.squeeze(0), log_prob

    def learn(self):
        """
        Perform REINFORCE update using stored memories.
        
        Returns:
            actor_loss: computed actor loss value
        """
        if len(self.memory.states) < self.batch_size:
            return None

        # Gather all stored values
        log_probs = torch.stack(self.memory.log_probs)  # shape [N]
        rewards = torch.tensor(self.memory.rewards, dtype=torch.float32, device=self.device)

        # Actor loss
        actor_loss = -(log_probs * rewards).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.actor_optimizer.step()

        self.memory.clear_memory()
        return actor_loss.item()
