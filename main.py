"""
Main Entry Point & Project Controller: Treasure Hunt Using Reinforcement Learning.
Provides interactive menu and command-line interfaces for training, evaluation, and GUI demo.
"""

import os
import sys
import argparse

# Safe stdout handling for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def banner():
    print("=" * 68)
    print("   TREASURE HUNT GAME USING REINFORCEMENT LEARNING (Q-LEARNING)")
    print("=" * 68)
    print("  An AI Agent that learns optimal navigation, obstacle avoidance,")
    print("  and treasure retrieval through trial-and-error reward feedback.")
    print("=" * 68)


def interactive_menu():
    while True:
        banner()
        print("\nChoose an option:")
        print("  [1] Launch Interactive Pygame Visual Interface")
        print("  [2] Train Q-Learning Agent (Generate Model & Learning Curves)")
        print("  [3] Run Evaluation Benchmark (Success Rate, Avg Steps, Trajectory)")
        print("  [4] Quick Retrain (Fast 400 Episodes)")
        print("  [5] Exit")
        print("-" * 68)

        choice = input("Enter your choice (1-5): ").strip()

        if choice == "1":
            print("\nLaunching Pygame GUI...")
            from game_gui import TreasureHuntGUI
            gui = TreasureHuntGUI()
            gui.run()
        elif choice == "2":
            ep_input = input("\nEnter number of episodes to train (default: 800): ").strip()
            episodes = int(ep_input) if ep_input.isdigit() else 800
            from train import train
            train(episodes=episodes)
            input("\nPress Enter to return to menu...")
        elif choice == "3":
            from evaluate import evaluate_agent
            evaluate_agent()
            input("\nPress Enter to return to menu...")
        elif choice == "4":
            from train import train
            train(episodes=400)
            input("\nPress Enter to return to menu...")
        elif choice == "5":
            print("\nExiting. Thank you!")
            sys.exit(0)
        else:
            print("\n[!] Invalid selection. Please enter a number between 1 and 5.")


def main():
    parser = argparse.ArgumentParser(description="Treasure Hunt RL Project Manager")
    parser.add_argument("--gui", action="store_true", help="Launch interactive Pygame GUI")
    parser.add_argument("--train", action="store_true", help="Train Q-learning agent")
    parser.add_argument("--eval", action="store_true", help="Evaluate trained agent")
    parser.add_argument("--episodes", type=int, default=800, help="Number of training episodes")
    args = parser.parse_args()

    if args.gui:
        from game_gui import TreasureHuntGUI
        gui = TreasureHuntGUI()
        gui.run()
    elif args.train:
        from train import train
        train(episodes=args.episodes)
    elif args.eval:
        from evaluate import evaluate_agent
        evaluate_agent()
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
