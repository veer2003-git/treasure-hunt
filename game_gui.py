"""
Interactive Pygame GUI for the Treasure Hunt Reinforcement Learning Project.
Features:
  - Watch Trained AI Navigate
  - Step-by-Step AI Execution
  - Interactive Human Play Mode
  - Q-Value Heatmap & Optimal Policy Arrows Overlay
  - Real-time Q-values HUD and statistics
"""

import os
import sys
import pygame
import numpy as np

from environment import TreasureHuntEnv, CellType, Action
from agent import QLearningAgent

# Colors (Modern Dark / Adventure Palette)
COLOR_BG = (22, 27, 34)           # Deep slate dark
COLOR_GRID_BG = (33, 38, 45)      # Card dark
COLOR_GRID_LINE = (48, 54, 61)    # Subtle border
COLOR_TEXT = (240, 246, 252)      # Bright text
COLOR_TEXT_MUTED = (139, 148, 158)# Subdued text
COLOR_SIDEBAR = (26, 32, 40)      # Sidebar panel

# Game Entity Colors
COLOR_WALL = (65, 75, 86)         # Solid wall block
COLOR_WALL_INNER = (50, 58, 68)
COLOR_TRAP = (218, 54, 51)        # Trap Red
COLOR_TRAP_GLOW = (110, 25, 25)
COLOR_TREASURE = (230, 180, 34)   # Gold
COLOR_TREASURE_GLOW = (255, 215, 0)
COLOR_AGENT = (46, 160, 255)      # Cyan / Neon Blue
COLOR_AGENT_TRAIL = (30, 110, 190, 90)
COLOR_POLICY_ARROW = (56, 189, 248) # Sky blue arrows

# Button Colors
COLOR_BTN_DEFAULT = (45, 55, 72)
COLOR_BTN_HOVER = (66, 80, 102)
COLOR_BTN_ACTIVE = (37, 99, 235)


class Button:
    """Clickable UI Button for Pygame."""
    def __init__(self, rect, text, callback, font, color=COLOR_BTN_DEFAULT, hover_color=COLOR_BTN_HOVER):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.font = font
        self.base_color = color
        self.hover_color = hover_color
        self.is_hovered = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()
                return True
        return False

    def draw(self, surface):
        col = self.hover_color if self.is_hovered else self.base_color
        pygame.draw.rect(surface, col, self.rect, border_radius=6)
        pygame.draw.rect(surface, (80, 95, 115), self.rect, width=1, border_radius=6)
        txt_surf = self.font.render(self.text, True, COLOR_TEXT)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        surface.blit(txt_surf, txt_rect)


class TreasureHuntGUI:
    """Main Graphical User Interface for the Treasure Hunt RL Game."""

    def __init__(self, model_path="assets/q_table.npy"):
        pygame.init()
        pygame.font.init()

        # Window & Layout Dimensions
        self.cell_size = 68
        self.env = TreasureHuntEnv()
        self.grid_rows = self.env.rows
        self.grid_cols = self.env.cols

        self.grid_width = self.grid_cols * self.cell_size
        self.grid_height = self.grid_rows * self.cell_size
        self.sidebar_width = 360
        self.margin = 25

        self.screen_width = self.grid_width + self.sidebar_width + (self.margin * 3)
        self.screen_height = max(self.grid_height + (self.margin * 2), 650)

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Treasure Hunt AI - Reinforcement Learning (Q-Learning)")

        self.clock = pygame.time.Clock()

        # Fonts
        self.font_title = pygame.font.SysFont("Segoe UI", 20, bold=True)
        self.font_heading = pygame.font.SysFont("Segoe UI", 16, bold=True)
        self.font_body = pygame.font.SysFont("Segoe UI", 13)
        self.font_body_bold = pygame.font.SysFont("Segoe UI", 13, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI", 11)

        # Agent & Model
        self.model_path = model_path
        self.agent = QLearningAgent(rows=self.grid_rows, cols=self.grid_cols)
        self.model_loaded = False
        if os.path.exists(self.model_path):
            try:
                self.agent.load(self.model_path)
                self.model_loaded = True
            except Exception as e:
                print(f"[WARN] Failed to load Q-table: {e}")

        # Game State
        self.mode = "AI"  # "AI" or "HUMAN"
        self.ai_running = False
        self.ai_delay_ms = 250  # Default delay between AI steps
        self.speed_levels = [("Slow", 500), ("Medium", 250), ("Fast", 80), ("Instant", 15)]
        self.speed_idx = 1
        self.last_step_time = 0

        self.show_policy_overlay = True
        self.show_heatmap_overlay = False

        self.state = self.env.reset()
        self.episode_reward = 0.0
        self.step_count = 0
        self.status_message = "Ready. Press [Run AI] or [Step AI]."
        self.status_color = (74, 222, 128)
        self.path_history = [self.state]

        # Setup Buttons
        self._setup_buttons()

    def _setup_buttons(self):
        """Constructs UI buttons positioned along the sidebar."""
        bx = self.grid_width + (self.margin * 2)
        by = 370
        bw = (self.sidebar_width - 10) // 2
        bh = 34
        gap_y = 42

        self.buttons = [
            Button((bx, by, bw, bh), "Run AI", self.toggle_ai_run, self.font_body_bold, color=(34, 139, 34)),
            Button((bx + bw + 10, by, bw, bh), "Step AI", self.step_ai_once, self.font_body_bold),
            
            Button((bx, by + gap_y, bw, bh), "Reset Game", self.reset_game, self.font_body_bold),
            Button((bx + bw + 10, by + gap_y, bw, bh), "Toggle Mode", self.toggle_mode, self.font_body_bold),

            Button((bx, by + gap_y * 2, bw, bh), "Policy: ON", self.toggle_policy, self.font_body),
            Button((bx + bw + 10, by + gap_y * 2, bw, bh), "Heatmap: OFF", self.toggle_heatmap, self.font_body),

            Button((bx, by + gap_y * 3, self.sidebar_width, bh), f"Speed: {self.speed_levels[self.speed_idx][0]}", self.cycle_speed, self.font_body),
        ]

    def reset_game(self):
        """Resets the environment and agent position."""
        self.state = self.env.reset()
        self.episode_reward = 0.0
        self.step_count = 0
        self.path_history = [self.state]
        self.status_message = "Reset. Agent at Start."
        self.status_color = (139, 148, 158)
        self.ai_running = False
        self.buttons[0].text = "Run AI"
        self.buttons[0].base_color = (34, 139, 34)

    def toggle_ai_run(self):
        """Toggles continuous AI play."""
        if not self.model_loaded:
            self.status_message = "Train model first (python train.py)"
            self.status_color = COLOR_TRAP
            return

        self.mode = "AI"
        self.ai_running = not self.ai_running
        if self.ai_running:
            self.buttons[0].text = "Pause AI"
            self.buttons[0].base_color = (180, 83, 9)
            self.status_message = "AI is navigating..."
            self.status_color = (56, 189, 248)
        else:
            self.buttons[0].text = "Run AI"
            self.buttons[0].base_color = (34, 139, 34)
            self.status_message = "AI paused."
            self.status_color = COLOR_TEXT_MUTED

    def step_ai_once(self):
        """Executes a single step using the learned policy."""
        if not self.model_loaded:
            self.status_message = "Train model first (python train.py)"
            self.status_color = COLOR_TRAP
            return

        self.mode = "AI"
        self._execute_agent_step()

    def toggle_mode(self):
        """Switches between AI autonomous mode and Human keyboard play."""
        self.ai_running = False
        self.buttons[0].text = "Run AI"
        self.buttons[0].base_color = (34, 139, 34)
        if self.mode == "AI":
            self.mode = "HUMAN"
            self.status_message = "Human Mode: Use Arrow Keys!"
            self.status_color = (250, 204, 21)
        else:
            self.mode = "AI"
            self.status_message = "AI Mode Active."
            self.status_color = (56, 189, 248)

    def toggle_policy(self):
        """Toggles display of policy arrows."""
        self.show_policy_overlay = not self.show_policy_overlay
        self.buttons[4].text = f"Policy: {'ON' if self.show_policy_overlay else 'OFF'}"

    def toggle_heatmap(self):
        """Toggles display of state value heatmap."""
        self.show_heatmap_overlay = not self.show_heatmap_overlay
        self.buttons[5].text = f"Heatmap: {'ON' if self.show_heatmap_overlay else 'OFF'}"

    def cycle_speed(self):
        """Cycles through simulation speeds."""
        self.speed_idx = (self.speed_idx + 1) % len(self.speed_levels)
        name, delay = self.speed_levels[self.speed_idx]
        self.ai_delay_ms = delay
        self.buttons[6].text = f"Speed: {name}"

    def _execute_agent_step(self):
        """Executes one step in the environment according to the agent's greedy policy."""
        action = self.agent.choose_action(self.state, greedy=True)
        next_state, reward, done, info = self.env.step(action)
        self.state = next_state
        self.episode_reward += reward
        self.step_count += 1
        self.path_history.append(self.state)

        event = info.get("event")
        if event == "found_treasure":
            self.status_message = f"TREASURE REACHED! (+{self.env.reward_treasure:.0f})"
            self.status_color = COLOR_TREASURE_GLOW
            self.ai_running = False
            self.buttons[0].text = "Run AI"
            self.buttons[0].base_color = (34, 139, 34)
        elif event == "hit_trap":
            self.status_message = f"FELL INTO TRAP! ({self.env.reward_trap:.0f})"
            self.status_color = COLOR_TRAP
            self.ai_running = False
            self.buttons[0].text = "Run AI"
            self.buttons[0].base_color = (34, 139, 34)
        elif event == "timeout":
            self.status_message = "Step budget exhausted (Timeout)"
            self.status_color = COLOR_TEXT_MUTED
            self.ai_running = False
            self.buttons[0].text = "Run AI"
            self.buttons[0].base_color = (34, 139, 34)
        else:
            self.status_message = f"Step {self.step_count}: Moved {self.env.ACTION_NAMES[Action(action)]}"
            self.status_color = (56, 189, 248)

    def handle_human_input(self, key):
        """Handles arrow keys when in Human Play mode."""
        if self.mode != "HUMAN":
            return

        action_map = {
            pygame.K_UP: Action.UP,
            pygame.K_DOWN: Action.DOWN,
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
        }

        if key in action_map:
            action = action_map[key]
            next_state, reward, done, info = self.env.step(action)
            self.state = next_state
            self.episode_reward += reward
            self.step_count += 1
            self.path_history.append(self.state)

            event = info.get("event")
            if event == "found_treasure":
                self.status_message = "YOU WON! Treasure Found!"
                self.status_color = COLOR_TREASURE_GLOW
            elif event == "hit_trap":
                self.status_message = "YOU DIED! Stepped on Trap!"
                self.status_color = COLOR_TRAP
            elif event.startswith("bump"):
                self.status_message = "Wall Collision! (-10)"
                self.status_color = (248, 113, 113)
            else:
                self.status_message = f"Human Move: {self.env.ACTION_NAMES[action]}"
                self.status_color = (250, 204, 21)

    def draw_grid(self):
        """Renders the game board, cells, traps, walls, treasure, and overlays."""
        gx = self.margin
        gy = self.margin

        # Value heatmap normalization if enabled
        if self.show_heatmap_overlay and self.model_loaded:
            v_table = np.max(self.agent.q_table, axis=2)
            v_min, v_max = np.min(v_table), np.max(v_table)
            v_range = max(1e-5, v_max - v_min)

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                rect = pygame.Rect(gx + c * self.cell_size, gy + r * self.cell_size, self.cell_size, self.cell_size)
                cell_type = CellType(self.env.grid[r, c])

                # Background color for cell
                if self.show_heatmap_overlay and self.model_loaded and cell_type != CellType.WALL:
                    norm = (v_table[r, c] - v_min) / v_range
                    # Color ramp from navy/blue to gold/yellow
                    cell_col = (
                        int(20 + 210 * norm),
                        int(30 + 170 * norm),
                        int(80 * (1 - norm) + 20),
                    )
                else:
                    cell_col = COLOR_GRID_BG

                pygame.draw.rect(self.screen, cell_col, rect)
                pygame.draw.rect(self.screen, COLOR_GRID_LINE, rect, 1)

                # Cell contents
                center = rect.center
                if cell_type == CellType.WALL:
                    # Impassable wall block
                    inner_rect = rect.inflate(-6, -6)
                    pygame.draw.rect(self.screen, COLOR_WALL, inner_rect, border_radius=5)
                    pygame.draw.rect(self.screen, COLOR_WALL_INNER, inner_rect, 2, border_radius=5)
                    # Brick pattern lines
                    pygame.draw.line(self.screen, COLOR_WALL_INNER, (inner_rect.left, center[1]), (inner_rect.right, center[1]), 2)
                elif cell_type == CellType.TRAP:
                    # Trap marker (Cross/Spike hazard)
                    pygame.draw.circle(self.screen, COLOR_TRAP_GLOW, center, self.cell_size // 3)
                    pygame.draw.circle(self.screen, COLOR_TRAP, center, self.cell_size // 4)
                    # X symbol
                    d = 9
                    pygame.draw.line(self.screen, COLOR_TEXT, (center[0] - d, center[1] - d), (center[0] + d, center[1] + d), 3)
                    pygame.draw.line(self.screen, COLOR_TEXT, (center[0] + d, center[1] - d), (center[0] - d, center[1] + d), 3)
                elif cell_type == CellType.TREASURE:
                    # Glowing golden chest / diamond
                    pygame.draw.circle(self.screen, (120, 95, 20), center, self.cell_size // 3)
                    pts = [
                        (center[0], center[1] - 16),
                        (center[0] + 16, center[1]),
                        (center[0], center[1] + 16),
                        (center[0] - 16, center[1]),
                    ]
                    pygame.draw.polygon(self.screen, COLOR_TREASURE, pts)
                    pygame.draw.polygon(self.screen, COLOR_TREASURE_GLOW, pts, 2)
                    # Crown/star core
                    pygame.draw.circle(self.screen, (255, 255, 255), center, 4)
                elif (r, c) == self.env.start_pos:
                    # Start Flag / S label
                    txt = self.font_body_bold.render("START", True, (100, 180, 255))
                    self.screen.blit(txt, txt.get_rect(center=center))

                # Optimal policy arrow overlay
                if (
                    self.show_policy_overlay
                    and self.model_loaded
                    and cell_type not in (CellType.WALL, CellType.TREASURE)
                ):
                    self._draw_policy_arrow(r, c, rect)

        # Draw path trail
        if len(self.path_history) > 1:
            points = [
                (gx + c * self.cell_size + self.cell_size // 2, gy + r * self.cell_size + self.cell_size // 2)
                for r, c in self.path_history
            ]
            pygame.draw.lines(self.screen, (56, 189, 248), False, points, 3)
            for pt in points[:-1]:
                pygame.draw.circle(self.screen, (56, 189, 248), pt, 4)

        # Draw Agent at Current Position
        agent_r, agent_c = self.state
        agent_center = (
            gx + agent_c * self.cell_size + self.cell_size // 2,
            gy + agent_r * self.cell_size + self.cell_size // 2,
        )
        # Outer glow
        pygame.draw.circle(self.screen, (30, 110, 190), agent_center, self.cell_size // 3)
        # Inner avatar
        pygame.draw.circle(self.screen, COLOR_AGENT, agent_center, self.cell_size // 4)
        pygame.draw.circle(self.screen, (255, 255, 255), agent_center, self.cell_size // 7)

    def _draw_policy_arrow(self, r, c, rect):
        """Draws small arrow indicating learned optimal action at cell (r, c)."""
        best_act = np.argmax(self.agent.q_table[r, c])
        max_q = np.max(self.agent.q_table[r, c])
        if abs(max_q) < 1e-4:
            return  # State not yet explored or all zeros

        cx, cy = rect.center
        sz = 8
        arrow_map = {
            Action.UP: [(cx, cy - sz), (cx - sz // 2, cy + sz // 2), (cx + sz // 2, cy + sz // 2)],
            Action.DOWN: [(cx, cy + sz), (cx - sz // 2, cy - sz // 2), (cx + sz // 2, cy - sz // 2)],
            Action.LEFT: [(cx - sz, cy), (cx + sz // 2, cy - sz // 2), (cx + sz // 2, cy + sz // 2)],
            Action.RIGHT: [(cx + sz, cy), (cx - sz // 2, cy - sz // 2), (cx - sz // 2, cy + sz // 2)],
        }
        pts = arrow_map.get(Action(best_act))
        if pts:
            pygame.draw.polygon(self.screen, COLOR_POLICY_ARROW, pts)

    def draw_sidebar(self):
        """Draws statistics panel, HUD, and live Q-values."""
        sx = self.grid_width + (self.margin * 2)
        sy = self.margin

        # Panel container
        panel_rect = pygame.Rect(sx - 10, sy, self.sidebar_width + 20, self.screen_height - (self.margin * 2))
        pygame.draw.rect(self.screen, COLOR_SIDEBAR, panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, COLOR_GRID_LINE, panel_rect, 1, border_radius=10)

        # Header Title
        title_surf = self.font_title.render("Treasure Hunt AI", True, (250, 204, 21))
        sub_surf = self.font_small.render("Reinforcement Learning (Q-Learning MDP)", True, COLOR_TEXT_MUTED)
        self.screen.blit(title_surf, (sx, sy + 15))
        self.screen.blit(sub_surf, (sx, sy + 42))

        # Status Banner
        pygame.draw.rect(self.screen, (33, 38, 45), (sx, sy + 68, self.sidebar_width, 36), border_radius=6)
        msg_surf = self.font_body_bold.render(self.status_message, True, self.status_color)
        self.screen.blit(msg_surf, (sx + 10, sy + 76))

        # Statistics HUD
        stats_y = sy + 118
        lines = [
            ("Operating Mode:", f"{self.mode} {'(Autonomous)' if self.mode == 'AI' else '(Arrow Keys)'}"),
            ("Agent Position (s):", f"Row: {self.state[0]}, Col: {self.state[1]}"),
            ("Steps Taken:", f"{self.step_count} / {self.env.max_steps}"),
            ("Accumulated Reward:", f"{self.episode_reward:+.1f}"),
            ("Q-Table Model:", "Loaded (assets/q_table.npy)" if self.model_loaded else "NOT FOUND!"),
        ]

        for i, (label, val) in enumerate(lines):
            lbl_surf = self.font_body.render(label, True, COLOR_TEXT_MUTED)
            val_surf = self.font_body_bold.render(val, True, COLOR_TEXT)
            self.screen.blit(lbl_surf, (sx, stats_y + i * 24))
            self.screen.blit(val_surf, (sx + 145, stats_y + i * 24))

        # Real-time Q-values at current position
        q_y = stats_y + (len(lines) * 24) + 12
        pygame.draw.line(self.screen, COLOR_GRID_LINE, (sx, q_y), (sx + self.sidebar_width, q_y), 1)

        q_head = self.font_heading.render(f"Q(s, a) Values at ({self.state[0]}, {self.state[1]}):", True, COLOR_TEXT)
        self.screen.blit(q_head, (sx, q_y + 8))

        if self.model_loaded:
            q_vals = self.agent.get_q_values(self.state)
            best_a = int(np.argmax(q_vals))
            actions_list = [("UP", 0), ("DOWN", 1), ("LEFT", 2), ("RIGHT", 3)]

            for idx, (name, a_code) in enumerate(actions_list):
                val = q_vals[a_code]
                is_best = (a_code == best_a)
                color = (74, 222, 128) if is_best else COLOR_TEXT_MUTED
                star = " ★ BEST" if is_best else ""
                q_text = self.font_body.render(f"  {name:5s}:  {val:8.2f}{star}", True, color)
                self.screen.blit(q_text, (sx, q_y + 34 + (idx * 20)))
        else:
            no_q = self.font_body.render("  Run training to populate Q-values", True, COLOR_TRAP)
            self.screen.blit(no_q, (sx, q_y + 34))

        # Draw Buttons
        for btn in self.buttons:
            btn.draw(self.screen)

        # Keyboard shortcuts hint
        hint_y = self.screen_height - self.margin - 40
        hint1 = self.font_small.render("Shortcuts: [Space] Run/Pause | [S] Step | [R] Reset", True, COLOR_TEXT_MUTED)
        hint2 = self.font_small.render("           [M] Mode | [P] Policy | [H] Heatmap", True, COLOR_TEXT_MUTED)
        self.screen.blit(hint1, (sx, hint_y))
        self.screen.blit(hint2, (sx, hint_y + 16))

    def run(self):
        """Main game loop."""
        running = True
        while running:
            current_time = pygame.time.get_ticks()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.toggle_ai_run()
                    elif event.key == pygame.K_s:
                        self.step_ai_once()
                    elif event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_m:
                        self.toggle_mode()
                    elif event.key == pygame.K_p:
                        self.toggle_policy()
                    elif event.key == pygame.K_h:
                        self.toggle_heatmap()
                    else:
                        self.handle_human_input(event.key)

                # Button clicks
                for btn in self.buttons:
                    btn.handle_event(event)

            # Continuous AI step timer
            if self.ai_running and (current_time - self.last_step_time >= self.ai_delay_ms):
                self._execute_agent_step()
                self.last_step_time = current_time

            # Render
            self.screen.fill(COLOR_BG)
            self.draw_grid()
            self.draw_sidebar()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    gui = TreasureHuntGUI()
    gui.run()
