# 🧭 Treasure Hunt Game Using Reinforcement Learning (Q-Learning)

An intelligent, interactive AI system where an agent learns through trial-and-error to navigate a hazardous grid environment, avoid deadly traps and impassable walls, and discover the optimal path to the hidden treasure.

Built with Python, NumPy, Matplotlib, and an interactive Pygame graphical interface.

---

## 📌 Project Overview & Reinforcement Learning Architecture

The system is formulated as a **Markov Decision Process (MDP)**:

$$\text{MDP} = \langle S, A, P, R, \gamma \rangle$$

### 1. State Space ($S$)
- Defined as the agent's coordinate $(r, c)$ in an $8 \times 8$ GridWorld ($64$ discrete states).
- **Start Position**: $(0, 0)$
- **Treasure Position**: $(7, 7)$
- Grid cells include: **Empty Paths**, **Obstacle Walls**, **Deadly Traps**, and the **Golden Treasure**.

### 2. Action Space ($A$)
Four discrete directional actions:
- `0`: **UP** $(-1, 0)$
- `1`: **DOWN** $(+1, 0)$
- `2`: **LEFT** $(0, -1)$
- `3`: **RIGHT** $(0, +1)$

### 3. Reward System ($R$)
The reward function drives the agent's behavior:
- 🏆 **Reach Treasure**: $+100.0$ (High positive reinforcement)
- 💀 **Fall into Trap**: $-50.0$ (Severe penalty and terminal episode end)
- 🧱 **Bump into Wall / Boundary**: $-10.0$ (Penalty; agent stays in current state)
- ⏱️ **Each Regular Step**: $-1.0$ (Living penalty that penalizes unnecessary loops and incentivizes the shortest path)

### 4. Q-Learning Algorithm & Bellman Equation
The agent updates its action-value function $Q(s, a)$ iteratively using the **Bellman Optimality Equation**:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

Where:
- $\alpha$ (**Learning Rate**): $0.1$
- $\gamma$ (**Discount Factor**): $0.95$ (prioritizes long-term future return)
- $r$: Immediate reward received
- $s'$: Next state reached
- $Q(s', a')$: Estimated maximum future value

### 5. Exploration vs. Exploitation ($\epsilon$-Greedy Strategy)
- The agent starts with $\epsilon = 1.0$ (100% random exploration to discover the map).
- Epsilon decays multiplicatively after each episode: $\epsilon \leftarrow \max(\epsilon_{min}, \epsilon \cdot \text{decay})$.
- After training, $\epsilon \approx 0.01$, and the agent exploits its learned knowledge greedily: $\pi(s) = \arg\max_a Q(s, a)$.

---

## 📁 Project Structure

```
project/
│
├── environment.py       # GridWorld environment (states, actions, rewards, transitions)
├── agent.py             # Q-Learning Agent (Q-table, Bellman update, epsilon-greedy)
├── train.py             # Training engine with convergence tracking & graph generation
├── evaluate.py          # Benchmark testing (Success Rate, Avg Steps, Trajectory)
├── game_gui.py          # Interactive Pygame graphical interface & HUD
├── main.py              # Central launcher (Interactive CLI & menu)
├── run.bat              # One-click Windows desktop launcher
├── requirements.txt     # Python dependencies
├── README.md            # Complete documentation & viva guide
└── assets/              # Stored model weights and evaluation plots
    ├── q_table.npy      # Pre-trained Q-table
    ├── q_table_meta.json# Model training configuration
    └── training_performance.png # Multi-panel learning curves
```

---

## 🚀 How to Run on Windows (Step-by-Step)

### Prerequisites
Make sure **Python 3.8+** is installed on your Windows PC and added to your `PATH`.

### Step 1: Open Terminal / Command Prompt
Open PowerShell or Command Prompt inside this folder:
```powershell
cd c:\Users\20252MCA0261\Desktop\project
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Application

#### Option A: One-Click Launch (Windows)
Double-click `run.bat` in File Explorer.

#### Option B: Interactive Main Menu
```powershell
python main.py
```
This opens an interactive menu to launch the GUI, train, or evaluate.

#### Option C: Directly Launch the Visual GUI
```powershell
python game_gui.py
```

#### Option D: Retrain the Agent
```powershell
python train.py --episodes 800
```
This trains the agent, saves the new `assets/q_table.npy`, and updates `assets/training_performance.png`.

#### Option E: Benchmark Evaluation
```powershell
python evaluate.py
```

---

## 🎮 Interactive Pygame GUI Features

The graphical interface is designed with a modern dark theme and rich HUD:

1. **Watch AI Run**: Click `[Run AI]` or press `[Space]` to watch the agent navigate autonomously.
2. **Step-by-Step Mode**: Click `[Step AI]` or press `[S]` to step one action at a time and inspect the decision-making process.
3. **Interactive Human Play**: Click `[Toggle Mode]` or press `[M]` and navigate manually using the **Arrow Keys** (Up, Down, Left, Right).
4. **Optimal Policy Arrows**: Toggle `[Policy: ON/OFF]` or press `[P]` to display directional arrows on every passable tile showing $\arg\max_a Q(s, a)$.
5. **State Value Heatmap**: Toggle `[Heatmap: ON/OFF]` or press `[H]` to visualize the learned value landscape $V(s) = \max_a Q(s, a)$.
6. **Speed Controller**: Toggle between Slow, Medium, Fast, and Instant playback.
7. **Real-time Q-Value HUD**: Displays live values for UP, DOWN, LEFT, RIGHT at the agent's current cell, highlighting the best action with a star `★ BEST`.

---

## 📊 Experimental Results & Benchmarks

Over 100 greedy evaluation test episodes ($\epsilon = 0$):
- **Success Rate (Treasure Found)**: **100.00%**
- **Trap Encounters**: **0.00%**
- **Optimal Path Length**: **14 steps**
- **Total Episode Reward**: **+87.00**
- **Convergence**: Achieved around episode 350-450.

### Learned Optimal Trajectory:
```
(0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) -> (2,3) -> (2,4) -> (3,4) 
      -> (3,5) -> (3,6) -> (4,6) -> (5,6) -> (5,7) -> (6,7) -> (7,7)
```
The agent discovers the shortest path through the maze while maintaining safe clearance from all hazard tiles.

---

## 🎓 Viva & Interview Preparation Guide

### Q1: What is Q-Learning?
**Answer:** Q-Learning is a model-free, off-policy temporal difference (TD) reinforcement learning algorithm that learns the quality (expected cumulative reward) of taking a specific action in a given state.

### Q2: Why is Q-Learning called "Off-Policy"?
**Answer:** Because the update rule uses the maximum action-value of the next state ($\max_{a'} Q(s', a')$) according to the greedy policy, regardless of what action was actually taken by the exploration policy ($\epsilon$-greedy).

### Q3: What is the purpose of the living penalty ($-1$ per step)?
**Answer:** Without a living penalty, the agent would only be rewarded at the end (+100) and might wander randomly in circles without penalty. The $-1$ penalty encourages finding the strictly shortest, most efficient route.

### Q4: How is the exploration vs. exploitation dilemma solved?
**Answer:** Through the $\epsilon$-greedy mechanism with exponential decay. Early on, $\epsilon$ is large ($1.0$) so the agent explores unknown cells. As training progresses, $\epsilon$ gradually decays so the agent begins exploiting known profitable paths.
