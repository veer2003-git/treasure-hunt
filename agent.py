"""
Q-Learning Agent Implementation for the Treasure Hunt Game.
Implements the Bellman optimality update, epsilon-greedy exploration, and policy extraction.
"""

import os
import json
import numpy as np
from environment import Action


class QLearningAgent:
    """
    Q-Learning Agent that maintains a tabular representation of action values Q(s, a).
    
    Bellman Update Equation:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
    """

    def __init__(
        self,
        rows=8,
        cols=8,
        num_actions=4,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    ):
        """
        Initialize the Q-learning agent.
        
        Args:
            rows (int): Grid row count.
            cols (int): Grid column count.
            num_actions (int): Number of actions (default 4).
            learning_rate (float): Step size alpha (0 < alpha <= 1).
            discount_factor (float): Future reward discount gamma (0 <= gamma <= 1).
            epsilon (float): Initial exploration probability.
            epsilon_min (float): Floor for exploration probability.
            epsilon_decay (float): Multiplicative decay factor per episode.
        """
        self.rows = rows
        self.cols = cols
        self.num_actions = num_actions

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table shaped as (rows, cols, num_actions)
        self.q_table = np.zeros((self.rows, self.cols, self.num_actions), dtype=np.float64)

    def choose_action(self, state, greedy=False):
        """
        Select an action using the epsilon-greedy strategy.
        
        Args:
            state (tuple): (row, col) coordinates.
            greedy (bool): If True, forces pure exploitation (ignores epsilon).
            
        Returns:
            int: Selected action index.
        """
        r, c = state

        # Exploration: pick a random action
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)

        # Exploitation: pick action with maximum Q-value for current state
        q_values = self.q_table[r, c]
        max_val = np.max(q_values)

        # Tie-breaking: if multiple actions have the identical maximum Q-value, pick randomly among them
        best_actions = np.where(q_values == max_val)[0]
        return int(np.random.choice(best_actions))

    def update(self, state, action, reward, next_state, done):
        """
        Updates the Q-value for (state, action) using the Bellman equation.
        
        Args:
            state (tuple): Current (row, col).
            action (int): Action taken.
            reward (float): Reward received.
            next_state (tuple): Resulting (row, col).
            done (bool): Whether the episode ended.
            
        Returns:
            float: Temporal Difference (TD) error.
        """
        r, c = state
        nr, nc = next_state

        current_q = self.q_table[r, c, action]

        if done:
            # Terminal state has no future expected return
            target_q = reward
        else:
            # Bellman backup: reward + discounted future best action
            best_future_q = np.max(self.q_table[nr, nc])
            target_q = reward + self.gamma * best_future_q

        # Temporal difference error
        td_error = target_q - current_q

        # Update Q-table
        self.q_table[r, c, action] += self.lr * td_error

        return float(td_error)

    def decay_epsilon(self):
        """Reduces epsilon by the decay factor until the minimum floor is reached."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_values(self, state):
        """Returns Q-values of all actions for a given state."""
        r, c = state
        return self.q_table[r, c].copy()

    def get_policy(self):
        """
        Extract the greedy policy pi(s) = argmax_a Q(s, a) for the entire grid.
        
        Returns:
            np.ndarray: 2D array of best action indices for each cell.
        """
        return np.argmax(self.q_table, axis=2)

    def save(self, filepath):
        """
        Saves the Q-table and agent configuration to disk.
        
        Args:
            filepath (str): Base file path (without or with .npy).
        """
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        if not filepath.endswith(".npy"):
            filepath += ".npy"

        np.save(filepath, self.q_table)

        # Also save metadata for reproducibility
        meta_filepath = filepath.replace(".npy", "_meta.json")
        meta = {
            "rows": self.rows,
            "cols": self.cols,
            "num_actions": self.num_actions,
            "learning_rate": self.lr,
            "discount_factor": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
        with open(meta_filepath, "w") as f:
            json.dump(meta, f, indent=4)

    def load(self, filepath):
        """
        Loads the Q-table from disk.
        
        Args:
            filepath (str): Path to saved .npy file.
        """
        if not filepath.endswith(".npy"):
            filepath += ".npy"

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Q-table file not found: {filepath}")

        self.q_table = np.load(filepath)
        self.rows, self.cols, self.num_actions = self.q_table.shape

        meta_filepath = filepath.replace(".npy", "_meta.json")
        if os.path.exists(meta_filepath):
            with open(meta_filepath, "r") as f:
                meta = json.load(f)
                self.lr = meta.get("learning_rate", self.lr)
                self.gamma = meta.get("discount_factor", self.gamma)
                self.epsilon_min = meta.get("epsilon_min", self.epsilon_min)
                self.epsilon_decay = meta.get("epsilon_decay", self.epsilon_decay)
                self.epsilon = self.epsilon_min  # When loaded, act greedily/evaluated
