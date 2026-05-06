"""
Reward calculation module for GUIDE framework.
Contains the get_Reward class for computing rewards based on BIC scores and cycle penalties.
"""

import numpy as np
import logging
import torch
from scipy.linalg import expm as matrix_exponential
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import pdist
from sklearn.gaussian_process import GaussianProcessRegressor as GPR


class get_Reward:
    """
    Reward calculator for causal DAG discovery.
    Computes rewards based on BIC scores, cycle penalties, and structural constraints.
    """
    
    _logger = logging.getLogger(__name__)

    def __init__(self, batch_num, maxlen, dim, inputdata, sl, su, lambda1_upper,
                 score_type='BIC', reg_type='LR', l1_graph_reg=0.0, verbose_flag=True):
        """
        Initialize the reward calculator.
        
        Args:
            batch_num: number of batches
            maxlen: maximum length (number of variables)
            dim: dimension parameter
            inputdata: input data matrix
            sl: lower bound for score transformation
            su: upper bound for score transformation
            lambda1_upper: upper bound for lambda1
            score_type: type of score ('BIC' or 'BIC_different_var')
            reg_type: type of regressor ('LR', 'QR', 'GPR')
            l1_graph_reg: L1 regularization for graph
            verbose_flag: whether to print verbose output
        """
        self.batch_num = batch_num
        self.maxlen = maxlen
        self.dim = dim
        self.baseint = 2**maxlen
        self.d = {}
        self.d_RSS = {}
        self.inputdata = inputdata
        self.n_samples = inputdata.shape[0]
        self.l1_graph_reg = l1_graph_reg
        self.verbose = verbose_flag
        self.sl = sl
        self.su = su
        self.lambda1_upper = lambda1_upper
        self.bic_penalty = np.log(inputdata.shape[0]) / inputdata.shape[0]

        if score_type not in ('BIC', 'BIC_different_var'):
            raise ValueError('Reward type not supported.')
        if reg_type not in ('LR', 'QR', 'GPR'):
            raise ValueError('Reg type not supported')
        self.score_type = score_type
        self.reg_type = reg_type

        self.ones = np.ones((inputdata.shape[0], 1), dtype=np.float32)
        self.poly = PolynomialFeatures()

    def cal_rewards(self, graphs, true_graph, lambda1, lambda2, lambda3):
        """
        Calculate rewards for a batch of graphs.
        
        Args:
            graphs: list of adjacency matrices
            true_graph: true adjacency matrix
            lambda1: penalty weight for cycle existence
            lambda2: penalty weight for cycle magnitude
            lambda3: penalty weight for structural mismatch
            
        Returns:
            rewards_batches: array of rewards
        """
        rewards_batches = []
        for graphi in graphs:
            reward_ = self.calculate_reward_single_graph(graphi, true_graph, lambda1, lambda2, lambda3)
            rewards_batches.append(reward_)
        return np.array(rewards_batches)

    def calculate_yerr(self, X_train, y_train):
        """
        Calculate prediction error using specified regressor.
        
        Args:
            X_train: training features
            y_train: training targets
            
        Returns:
            y_err: prediction errors
        """
        if self.reg_type == 'LR':
            return self.calculate_LR(X_train, y_train)
        elif self.reg_type == 'QR':
            return self.calculate_QR(X_train, y_train)
        elif self.reg_type == 'GPR':
            return self.calculate_GPR(X_train, y_train)
        else:
            assert False, 'Regressor not supported'

    def calculate_LR(self, X_train, y_train):
        """Calculate prediction error using Linear Regression."""
        X = np.hstack((X_train, self.ones))
        XtX = X.T.dot(X)
        Xty = X.T.dot(y_train)
        theta = np.linalg.solve(XtX, Xty)
        y_err = X.dot(theta) - y_train
        return y_err

    def calculate_QR(self, X_train, y_train):
        """Calculate prediction error using Quadratic Regression."""
        X_train = self.poly.fit_transform(X_train)[:, 1:]
        return self.calculate_LR(X_train, y_train)

    def calculate_GPR(self, X_train, y_train):
        """Calculate prediction error using Gaussian Process Regression."""
        med_w = np.median(pdist(X_train, 'euclidean'))
        gpr = GPR().fit(X_train / med_w, y_train)
        return y_train.reshape(-1, 1) - gpr.predict(X_train / med_w).reshape(-1, 1)

    def calculate_reward_single_graph(self, graph_batch, tgraph, lambda1, lambda2, lambda3):
        """
        Calculate reward for a single graph.
        
        Args:
            graph_batch: adjacency matrix to evaluate
            tgraph: true adjacency matrix for comparison
            lambda1: penalty weight for cycle existence
            lambda2: penalty weight for cycle magnitude
            lambda3: penalty weight for structural mismatch
            
        Returns:
            reward: computed reward value
            score: BIC score
            cycness: cycle penalty
            penalty: structural mismatch penalty
        """
        if isinstance(graph_batch, torch.Tensor):
            graph_batch = graph_batch.cpu().detach().numpy()
        if isinstance(tgraph, torch.Tensor):
            tgraph = tgraph.cpu().detach().numpy()

        # Simple mismatch penalty
        penalty = 0
        for i in range(self.maxlen):
            for j in range(self.maxlen):
                if tgraph[i][j] in [0, 1]:  # known edge or known non-edge
                    if graph_batch[i][j] != tgraph[i][j]:
                        penalty += 1

        # Zero diagonal
        for i in range(self.maxlen):
            graph_batch[i][i] = 0

        graph_to_int = []
        graph_to_int2 = []
        for i in range(self.maxlen):
            tt = np.int32(graph_batch[i])
            graph_to_int2.append(int(''.join([str(ad) for ad in tt]), 2))
            graph_to_int.append(self.baseint * i + int(''.join([str(ad) for ad in tt]), 2))

        graph_batch_to_tuple = tuple(graph_to_int2)
        if graph_batch_to_tuple in self.d:
            score_cyc = self.d[graph_batch_to_tuple]
            return self.penalized_score(score_cyc, lambda1, lambda2, lambda3), score_cyc[0], score_cyc[1], score_cyc[2]

        # Compute RSS
        RSS_ls = []
        for i in range(self.maxlen):
            col = graph_batch[i]
            if graph_to_int[i] in self.d_RSS:
                RSS_ls.append(self.d_RSS[graph_to_int[i]])
                continue

            if np.sum(col) < 0.1:
                y_err = self.inputdata[:, i]
                y_err = y_err - np.mean(y_err)
            else:
                cols_TrueFalse = col > 0.5
                X_train = self.inputdata[:, cols_TrueFalse]
                y_train = self.inputdata[:, i]
                y_err = self.calculate_yerr(X_train, y_train)

            RSSi = np.sum(np.square(y_err))
            if self.reg_type == 'GPR':
                RSSi += 1.0
            RSS_ls.append(RSSi)
            self.d_RSS[graph_to_int[i]] = RSSi

        if self.score_type == 'BIC':
            BIC = np.log(np.sum(RSS_ls)/self.n_samples+1e-8) \
                  + np.sum(graph_batch)*self.bic_penalty/self.maxlen
        elif self.score_type == 'BIC_different_var':
            BIC = np.sum(np.log(np.array(RSS_ls)/self.n_samples+1e-8)) \
                  + np.sum(graph_batch)*self.bic_penalty

        score = self.score_transform(BIC)
        cycness = np.trace(matrix_exponential(np.array(graph_batch))) - self.maxlen
        reward = -(score + lambda1*float(cycness > 1e-5) + lambda2*cycness + lambda3*penalty)

        if self.l1_graph_reg > 0:
            reward += self.l1_graph_reg * np.sum(graph_batch)
            score  += self.l1_graph_reg * np.sum(graph_batch)

        self.d[graph_batch_to_tuple] = (score, cycness, penalty)
        return reward, score, cycness, penalty

    def score_transform(self, s):
        """Transform score to normalized range."""
        return (s - self.sl) / (self.su - self.sl) * self.lambda1_upper

    def penalized_score(self, score_cyc, lambda1, lambda2, lambda3):
        """Calculate penalized score."""
        score, cyc, penalty = score_cyc
        return -(score + lambda1*float(cyc>1e-5) + lambda2*cyc + lambda3*penalty)

    def update_scores(self, score_cycs, lambda1, lambda2, lambda3):
        """Update scores for a list of score-cycle tuples."""
        ls = []
        for score_cyc in score_cycs:
            ls.append(self.penalized_score(score_cyc, lambda1, lambda2, lambda3))
        return ls

    def update_all_scores(self, lambda1, lambda2, lambda3):
        """Update all stored scores with new penalty weights."""
        score_cycs = list(self.d.items())
        ls = []
        for graph_int, score_cyc in score_cycs:
            ls.append((
                graph_int,
                (
                    self.penalized_score(score_cyc, lambda1, lambda2, lambda3),
                    score_cyc[0],
                    score_cyc[1]
                )
            ))
        return sorted(ls, key=lambda x: x[1][0])
