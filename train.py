"""
Training Module for Treasure Hunt Reinforcement Learning Agent.
Runs the Q-learning loop, tracks convergence metrics, and generates visualization plots.
"""

import os
import sys
import argparse
import numpy as np

# Safe stdout handling for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib.pyplot as plt
from environment import TreasureHuntEnv, CellType
from agent import QLearningAgent


def train(
    episodes=800,
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon_start=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    max_steps=100,
    save_path="assets/q_table.npy",
    plot_path="assets/training_performance.png",
    verbose=True,
):
    """
    Trains the Q-Learning agent on the Treasure Hunt environment.
    
    Returns:
        dict: Training history including rewards, steps, and success rates.
    """
    env = TreasureHuntEnv(max_steps_per_episode=max_steps)
    agent = QLearningAgent(
        rows=env.rows,
        cols=env.cols,
        num_actions=env.get_action_space_size(),
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )

    rewards_history = []
    steps_history = []
    success_history = []
    epsilons_history = []

    print("=" * 60)
    print("[*] STARTING REINFORCEMENT LEARNING TRAINING: TREASURE HUNT")
    print("=" * 60)
    print(f"Grid Size:        {env.rows}x{env.cols}")
    print(f"Total Episodes:   {episodes}")
    print(f"Learning Rate:    {learning_rate}")
    print(f"Discount Factor:  {discount_factor}")
    print(f"Epsilon Decay:    {epsilon_decay} (Min: {epsilon_min})")
    print("-" * 60)

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        success = False

        while not done:
            action = agent.choose_action(state, greedy=False)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            if info.get("event") == "found_treasure":
                success = True

        agent.decay_epsilon()

        rewards_history.append(total_reward)
        steps_history.append(steps)
        success_history.append(1 if success else 0)
        epsilons_history.append(agent.epsilon)

        # Logging periodic milestones
        if verbose and (ep % 50 == 0 or ep == 1 or ep == episodes):
            recent_rewards = rewards_history[-50:]
            recent_success = success_history[-50:]
            avg_rew = np.mean(recent_rewards)
            win_rate = np.mean(recent_success) * 100
            avg_steps = np.mean(steps_history[-50:])
            print(
                f"Episode {ep:4d}/{episodes} | "
                f"Avg Reward (last 50): {avg_rew:7.2f} | "
                f"Win Rate: {win_rate:5.1f}% | "
                f"Avg Steps: {avg_steps:5.1f} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

    # Save trained model
    agent.save(save_path)
    print("-" * 60)
    print(f"[SUCCESS] Training completed successfully!")
    print(f"[SAVED] Trained Q-table saved to: {os.path.abspath(save_path)}")

    # Plot metrics
    history = {
        "rewards": rewards_history,
        "steps": steps_history,
        "success": success_history,
        "epsilons": epsilons_history,
    }
    plot_training_metrics(history, plot_path)
    print(f"[SAVED] Training graphs saved to: {os.path.abspath(plot_path)}")
    print("=" * 60)

    return agent, history


def plot_training_metrics(history, output_path="assets/training_performance.png"):
    """
    Creates informative graphs visualizing the RL learning process.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rewards = history["rewards"]
    steps = history["steps"]
    successes = history["success"]
    epsilons = history["epsilons"]
    episodes = len(rewards)

    # Window for rolling average
    window = min(50, episodes // 5 if episodes >= 20 else 1)
    
    def moving_avg(arr, w):
        if len(arr) < w:
            return arr
        return np.convolve(arr, np.ones(w) / w, mode="valid")

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    fig.suptitle("Treasure Hunt - Reinforcement Learning (Q-Learning) Performance", fontsize=15, weight="bold")

    # 1. Episode Rewards
    axes[0].plot(rewards, color="#90CAF9", alpha=0.4, label="Episode Reward")
    ma_rewards = moving_avg(rewards, window)
    axes[0].plot(range(window - 1, episodes), ma_rewards, color="#1976D2", linewidth=2.2, label=f"{window}-Ep Moving Avg")
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.6)
    axes[0].set_ylabel("Total Reward", fontsize=11, weight="bold")
    axes[0].set_title("Reward Convergence (Agent Learns to Avoid Traps & Claim Treasure)", fontsize=11)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # 2. Steps to Complete Episode
    axes[1].plot(steps, color="#FFCC80", alpha=0.4, label="Steps per Episode")
    ma_steps = moving_avg(steps, window)
    axes[1].plot(range(window - 1, episodes), ma_steps, color="#F57C00", linewidth=2.2, label=f"{window}-Ep Moving Avg")
    axes[1].set_ylabel("Steps Count", fontsize=11, weight="bold")
    axes[1].set_title("Path Optimization (Agent Discovers Shortest Path)", fontsize=11)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # 3. Epsilon Decay & Win Rate
    ax3_twin = axes[2].twinx()
    axes[2].plot(epsilons, color="#9C27B0", linestyle="-.", linewidth=1.8, label="Epsilon (Exploration Rate)")
    axes[2].set_ylabel("Epsilon ε", color="#9C27B0", fontsize=11, weight="bold")
    axes[2].tick_params(axis="y", labelcolor="#9C27B0")

    ma_success = moving_avg([s * 100 for s in successes], window)
    ax3_twin.plot(range(window - 1, episodes), ma_success, color="#2E7D32", linewidth=2.2, label=f"Win Rate % ({window}-Ep Avg)")
    ax3_twin.set_ylabel("Treasure Reached (%)", color="#2E7D32", fontsize=11, weight="bold")
    ax3_twin.tick_params(axis="y", labelcolor="#2E7D32")
    ax3_twin.set_ylim(-5, 105)

    axes[2].set_xlabel("Episode", fontsize=11, weight="bold")
    axes[2].set_title("Exploration vs Exploitation Tradeoff & Success Rate", fontsize=11)
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Q-Learning Agent for Treasure Hunt")
    parser.add_argument("--episodes", type=int, default=800, help="Total training episodes")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate alpha")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor gamma")
    parser.add_argument("--save", type=str, default="assets/q_table.npy", help="Save path for Q-table")
    parser.add_argument("--plot", type=str, default="assets/training_performance.png", help="Path for performance plot")
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        learning_rate=args.lr,
        discount_factor=args.gamma,
        save_path=args.save,
        plot_path=args.plot,
    )
