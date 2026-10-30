"""
Treasure Hunt GridWorld Environment for Reinforcement Learning.
Defines the state space, action space, reward structure, and transition dynamics.
"""

from enum import IntEnum
import numpy as np


class Action(IntEnum):
    """Discrete action space for navigation in 2D grid."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class CellType(IntEnum):
    """Types of cells available in the treasure hunt grid."""
    EMPTY = 0
    WALL = 1
    TRAP = 2
    TREASURE = 3
    START = 4


class TreasureHuntEnv:
    """
    GridWorld environment for the Treasure Hunt RL agent.
    
    State Representation:
        Tuple (row, col) representing the agent's current position on the grid.
        
    Action Space:
        0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        
    Reward Function:
        - Reaching Treasure: +100
        - Falling into Trap: -50 (terminal or penalty)
        - Hitting a Wall / Boundary: -10 (agent stays in place)
        - Every valid step: -1 (living penalty to encourage optimal shortest path)
    """

    # Action displacements: (delta_row, delta_col)
    ACTION_EFFECTS = {
        Action.UP: (-1, 0),
        Action.DOWN: (1, 0),
        Action.LEFT: (0, -1),
        Action.RIGHT: (0, 1),
    }

    ACTION_NAMES = {
        Action.UP: "UP",
        Action.DOWN: "DOWN",
        Action.LEFT: "LEFT",
        Action.RIGHT: "RIGHT",
    }

    def __init__(self, grid=None, max_steps_per_episode=100, trap_is_terminal=True):
        """
        Initialize the environment.
        
        Args:
            grid (list of list or np.ndarray, optional): 2D map defining the environment.
            max_steps_per_episode (int): Step budget before truncation.
            trap_is_terminal (bool): If True, stepping into a trap ends the episode.
        """
        if grid is None:
            self.grid = self.get_default_grid()
        else:
            self.grid = np.array(grid, dtype=np.int32)

        self.rows, self.cols = self.grid.shape
        self.max_steps = max_steps_per_episode
        self.trap_is_terminal = trap_is_terminal

        # Locate key positions
        self.start_pos = self._find_position(CellType.START)
        if self.start_pos is None:
            self.start_pos = (0, 0)
            self.grid[0, 0] = CellType.START

        self.treasure_pos = self._find_position(CellType.TREASURE)
        if self.treasure_pos is None:
            self.treasure_pos = (self.rows - 1, self.cols - 1)
            self.grid[self.rows - 1, self.cols - 1] = CellType.TREASURE

        # Rewards configuration
        self.reward_treasure = 100.0
        self.reward_trap = -50.0
        self.reward_wall = -10.0
        self.reward_step = -1.0

        # Current episode state
        self.agent_pos = self.start_pos
        self.current_step = 0

    @staticmethod
    def get_default_grid():
        """
        Creates a balanced 8x8 treasure hunt grid.
        0: Empty, 1: Wall, 2: Trap, 3: Treasure, 4: Start
        """
        # 0: Empty, 1: Wall (impassable), 2: Trap (danger), 3: Treasure, 4: Start
        return np.array([
            [4, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 2, 0, 0],
            [0, 1, 0, 0, 0, 1, 1, 0],
            [0, 0, 2, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 2, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 2, 1, 1, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 1, 0, 3],
        ], dtype=np.int32)

    def _find_position(self, cell_type):
        """Find row, col coordinates of a given cell type."""
        coords = np.argwhere(self.grid == cell_type)
        if len(coords) > 0:
            return int(coords[0][0]), int(coords[0][1])
        return None

    def reset(self):
        """
        Resets environment to starting state.
        
        Returns:
            tuple: (row, col) initial agent position
        """
        self.agent_pos = self.start_pos
        self.current_step = 0
        return self.agent_pos

    def step(self, action):
        """
        Execute an action in the environment.
        
        Args:
            action (int or Action): Action to execute (0..3)
            
        Returns:
            next_state (tuple): (row, col) new agent coordinates
            reward (float): Reward received from the transition
            done (bool): Whether the episode has terminated
            info (dict): Diagnostic info (e.g., event summary, step count)
        """
        self.current_step += 1
        action = Action(action)
        delta_r, delta_c = self.ACTION_EFFECTS[action]

        target_r = self.agent_pos[0] + delta_r
        target_c = self.agent_pos[1] + delta_c

        done = False
        event = "step"

        # Check grid boundaries
        if not (0 <= target_r < self.rows and 0 <= target_c < self.cols):
            # Bumped into grid boundary
            reward = self.reward_wall
            next_pos = self.agent_pos
            event = "bump_boundary"
        elif self.grid[target_r, target_c] == CellType.WALL:
            # Bumped into internal wall
            reward = self.reward_wall
            next_pos = self.agent_pos
            event = "bump_wall"
        else:
            # Move to new position
            next_pos = (target_r, target_c)
            cell = self.grid[target_r, target_c]

            if cell == CellType.TREASURE:
                reward = self.reward_treasure
                done = True
                event = "found_treasure"
            elif cell == CellType.TRAP:
                reward = self.reward_trap
                if self.trap_is_terminal:
                    done = True
                event = "hit_trap"
            else:
                # Normal move
                reward = self.reward_step
                event = "move"

        self.agent_pos = next_pos

        # Check maximum steps truncation
        if self.current_step >= self.max_steps and not done:
            done = True
            event = "timeout"

        info = {
            "event": event,
            "step": self.current_step,
            "position": self.agent_pos,
        }

        return self.agent_pos, reward, done, info

    def get_state_space_size(self):
        """Returns total number of possible grid states."""
        return self.rows * self.cols

    def get_action_space_size(self):
        """Returns number of discrete actions."""
        return len(Action)

    def state_to_index(self, state):
        """Maps (row, col) tuple to a 1D scalar index."""
        r, c = state
        return r * self.cols + c

    def index_to_state(self, index):
        """Maps 1D scalar index back to (row, col) coordinates."""
        return divmod(index, self.cols)
