#!/usr/bin/env python3
"""
Compare ARPO and AEPO Training Results

This script compares training logs and metrics between ARPO and AEPO models.
"""

import argparse
import re
from pathlib import Path


def parse_log_file(log_file):
    """
    Parse training log file and extract metrics.

    Args:
        log_file: Path to training log

    Returns:
        Dictionary with rewards and losses
    """
    rewards = []
    losses = []
    steps = []

    if not Path(log_file).exists():
        print(f"❌ Log file not found: {log_file}")
        return None

    with open(log_file, 'r') as f:
        for line in f:
            # Match lines like: Epoch X | Step Y | Reward: 0.XX | Loss: 0.XX
            if "Reward:" in line and "Loss:" in line:
                try:
                    # Extract step
                    step_match = re.search(r'Step (\d+)', line)
                    if step_match:
                        step = int(step_match.group(1))
                        steps.append(step)

                    # Extract reward
                    reward_match = re.search(r'Reward:\s*([\d.]+)', line)
                    if reward_match:
                        reward = float(reward_match.group(1))
                        rewards.append(reward)

                    # Extract loss
                    loss_match = re.search(r'Loss:\s*([\d.]+)', line)
                    if loss_match:
                        loss = float(loss_match.group(1))
                        losses.append(loss)
                except Exception as e:
                    continue

    return {
        'steps': steps,
        'rewards': rewards,
        'losses': losses
    }


def print_summary(name, metrics):
    """Print summary statistics for a model."""
    if not metrics or not metrics['rewards']:
        print(f"\n❌ No data for {name}")
        return

    rewards = metrics['rewards']
    losses = metrics['losses']

    print(f"\n{name}:")
    print(f"{'─'*40}")
    print(f"  Total Steps:    {len(rewards)}")
    print(f"  Initial Reward: {rewards[0]:.4f}")
    print(f"  Final Reward:   {rewards[-1]:.4f}")
    print(f"  Max Reward:     {max(rewards):.4f}")
    print(f"  Avg Reward (last 10): {sum(rewards[-10:])/min(10, len(rewards)):.4f}")
    print(f"  ")
    print(f"  Initial Loss:   {losses[0]:.4f}")
    print(f"  Final Loss:     {losses[-1]:.4f}")
    print(f"  Min Loss:       {min(losses):.4f}")


def compare_models(arpo_log, aepo_log):
    """
    Compare ARPO and AEPO training results.

    Args:
        arpo_log: Path to ARPO training log
        aepo_log: Path to AEPO training log
    """
    print(f"\n{'='*60}")
    print(f"ARPO vs AEPO Training Comparison")
    print(f"{'='*60}")

    # Parse logs
    print(f"\n📂 Parsing training logs...")
    arpo_metrics = parse_log_file(arpo_log)
    aepo_metrics = parse_log_file(aepo_log)

    if arpo_metrics is None and aepo_metrics is None:
        print(f"\n❌ No log files found. Please check paths.")
        return

    # Print summaries
    print(f"\n{'='*60}")
    print(f"Training Summary")
    print(f"{'='*60}")

    if arpo_metrics:
        print_summary("ARPO", arpo_metrics)
    if aepo_metrics:
        print_summary("AEPO", aepo_metrics)

    # Compare final rewards
    if arpo_metrics and aepo_metrics:
        print(f"\n{'='*60}")
        print(f"Head-to-Head Comparison")
        print(f"{'='*60}\n")

        arpo_final = arpo_metrics['rewards'][-1]
        aepo_final = aepo_metrics['rewards'][-1]

        print(f"Final Reward:")
        print(f"  ARPO: {arpo_final:.4f}")
        print(f"  AEPO: {aepo_final:.4f}")

        if aepo_final > arpo_final:
            diff = ((aepo_final - arpo_final) / arpo_final) * 100
            print(f"  Winner: AEPO (+{diff:.2f}%)")
        elif arpo_final > aepo_final:
            diff = ((arpo_final - aepo_final) / aepo_final) * 100
            print(f"  Winner: ARPO (+{diff:.2f}%)")
        else:
            print(f"  Result: Tie")

        # Compare avg reward (last 10 steps)
        arpo_avg = sum(arpo_metrics['rewards'][-10:]) / min(10, len(arpo_metrics['rewards']))
        aepo_avg = sum(aepo_metrics['rewards'][-10:]) / min(10, len(aepo_metrics['rewards']))

        print(f"\nAvg Reward (last 10 steps):")
        print(f"  ARPO: {arpo_avg:.4f}")
        print(f"  AEPO: {aepo_avg:.4f}")

        if aepo_avg > arpo_avg:
            diff = ((aepo_avg - arpo_avg) / arpo_avg) * 100
            print(f"  Winner: AEPO (+{diff:.2f}%)")
        elif arpo_avg > aepo_avg:
            diff = ((arpo_avg - aepo_avg) / aepo_avg) * 100
            print(f"  Winner: ARPO (+{diff:.2f}%)")
        else:
            print(f"  Result: Tie")

    # Optional: Create plot if matplotlib available
    try:
        import matplotlib.pyplot as plt

        if arpo_metrics and aepo_metrics:
            print(f"\n📊 Creating comparison plot...")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Plot rewards
            ax1.plot(arpo_metrics['steps'], arpo_metrics['rewards'],
                    label='ARPO', linewidth=2, marker='o', markersize=4)
            ax1.plot(aepo_metrics['steps'], aepo_metrics['rewards'],
                    label='AEPO', linewidth=2, marker='s', markersize=4)
            ax1.set_xlabel('Training Step', fontsize=12)
            ax1.set_ylabel('Reward', fontsize=12)
            ax1.set_title('Training Reward Comparison', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)

            # Plot losses
            ax2.plot(arpo_metrics['steps'], arpo_metrics['losses'],
                    label='ARPO', linewidth=2, marker='o', markersize=4)
            ax2.plot(aepo_metrics['steps'], aepo_metrics['losses'],
                    label='AEPO', linewidth=2, marker='s', markersize=4)
            ax2.set_xlabel('Training Step', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            output_file = 'arpo_aepo_comparison.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✅ Comparison plot saved to: {output_file}")
    except ImportError:
        print(f"\n📊 matplotlib not available - skipping plot generation")
        print(f"   Install with: pip install matplotlib")

    print(f"\n{'='*60}")
    print(f"✅ Comparison Complete!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare ARPO and AEPO training results"
    )
    parser.add_argument(
        '--arpo_log',
        default='/data/checkpoints/ARPO_Qwen2.5_7B_Reasoning/run.log',
        help='Path to ARPO training log'
    )
    parser.add_argument(
        '--aepo_log',
        default='/data/checkpoints/AEPO_Qwen2.5_7B_Reasoning/run.log',
        help='Path to AEPO training log'
    )

    args = parser.parse_args()

    compare_models(
        arpo_log=args.arpo_log,
        aepo_log=args.aepo_log
    )


if __name__ == "__main__":
    main()
