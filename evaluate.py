"""
Evaluation & Benchmarking Module for Trained Q-Learning Agent.
Tests the agent greedily (epsilon=0) to verify optimal path, success rate, and safety.
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

from environment import TreasureHuntEnv, CellType, Action
from agent import QLearningAgent


def evaluate_agent(
    model_path="assets/q_table.npy",
    test_episodes=100,
    max_steps=100,
    render_optimal_path=True,
):
    """
    Evaluates the agent using greedy policy (epsilon = 0).
    """
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at: {model_path}")
        print("Please train the agent first using: python train.py")
        return None

    env = TreasureHuntEnv(max_steps_per_episode=max_steps)
    agent = QLearningAgent(
        rows=env.rows,
        cols=env.cols,
        num_actions=env.get_action_space_size(),
    )
    agent.load(model_path)

    print("=" * 65)
    print("[*] EVALUATING TRAINED Q-LEARNING AGENT (GREEDY POLICY ε = 0)")
    print("=" * 65)
    print(f"Model File:       {os.path.abspath(model_path)}")
    print(f"Test Episodes:    {test_episodes}")
    print(f"Grid Size:        {env.rows}x{env.cols}")
    print("-" * 65)

    success_count = 0
    trap_count = 0
    timeout_count = 0
    total_steps_list = []
    total_rewards_list = []

    for ep in range(1, test_episodes + 1):
        state = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        outcome = "timeout"

        while not done:
            action = agent.choose_action(state, greedy=True)
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            steps += 1
            state = next_state

            if info.get("event") == "found_treasure":
                outcome = "success"
            elif info.get("event") == "hit_trap":
                outcome = "trap"

        if outcome == "success":
            success_count += 1
        elif outcome == "trap":
            trap_count += 1
        else:
            timeout_count += 1

        total_steps_list.append(steps)
        total_rewards_list.append(ep_reward)

    win_rate = (success_count / test_episodes) * 100
    trap_rate = (trap_count / test_episodes) * 100
    timeout_rate = (timeout_count / test_episodes) * 100
    avg_steps = np.mean(total_steps_list)
    avg_reward = np.mean(total_rewards_list)

    print("[SUMMARY BENCHMARK RESULTS]")
    print(f"  - Success Rate (Treasure Found):  {win_rate:6.2f}% ({success_count}/{test_episodes})")
    print(f"  - Trap Encounters:                {trap_rate:6.2f}% ({trap_count}/{test_episodes})")
    print(f"  - Timeouts / Incomplete:          {timeout_rate:6.2f}% ({timeout_count}/{test_episodes})")
    print(f"  - Average Steps to Treasure:      {avg_steps:6.2f}")
    print(f"  - Average Total Episode Reward:   {avg_reward:6.2f}")
    print("-" * 65)

    if render_optimal_path:
        trace_and_print_path(env, agent)

    print("=" * 65)
    return {
        "success_rate": win_rate,
        "trap_rate": trap_rate,
        "avg_steps": avg_steps,
        "avg_reward": avg_reward,
    }


def trace_and_print_path(env, agent):
    """
    Traces the greedy path from start to goal and displays an ASCII representation.
    """
    state = env.reset()
    done = False
    path = [state]
    actions_taken = []
    step_num = 0

    while not done and step_num < env.max_steps:
        action = agent.choose_action(state, greedy=True)
        actions_taken.append(env.ACTION_NAMES[Action(action)])
        next_state, reward, done, info = env.step(action)
        path.append(next_state)
        state = next_state
        step_num += 1

    print("\n[OPTIMAL LEARNED PATHWAY]")
    print(f"Path Length: {len(path) - 1} steps")
    print("Coordinates Visited:")
    coord_str = " -> ".join([f"({r},{c})" for r, c in path])
    print(f"  {coord_str}")
    print(f"Actions Executed: {' -> '.join(actions_taken)}")

    # Visual ASCII Grid of the Path
    symbols = {
        CellType.EMPTY: " . ",
        CellType.WALL: "###",
        CellType.TRAP: "[X]",
        CellType.TREASURE: "[$]",
        CellType.START: "[S]",
    }

    path_set = set(path[1:-1])  # Exclude start and treasure for path marker
    print("\n[GRID TRAJECTORY MAP]")
    print("Legend: [S]=Start, [$]=Treasure, ###=Wall, [X]=Trap, (*)=Learned Path,  . =Empty")
    print("   " + "".join([f"  {c} " for c in range(env.cols)]))

    for r in range(env.rows):
        row_str = f"{r:2d} "
        for c in range(env.cols):
            if (r, c) == env.start_pos:
                row_str += " S  "
            elif (r, c) == env.treasure_pos:
                row_str += " $  "
            elif (r, c) in path_set:
                row_str += "(*) "
            else:
                row_str += symbols[CellType(env.grid[r, c])] + " "
        print(row_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained Q-Learning Agent")
    parser.add_argument("--model", type=str, default="assets/q_table.npy", help="Path to trained Q-table")
    parser.add_argument("--episodes", type=int, default=100, help="Number of test episodes")
    args = parser.parse_args()

    evaluate_agent(model_path=args.model, test_episodes=args.episodes)
