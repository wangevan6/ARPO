#!/usr/bin/env python3
"""
Create Toy Dataset for Local RL Testing

This script creates a small subset of the full RL training dataset
for quick local testing and pipeline verification.
"""

import pandas as pd
import argparse
import sys
from pathlib import Path


def create_toy_dataset(input_file, output_file, num_samples=100):
    """
    Create a toy dataset with limited samples for testing.

    Args:
        input_file: Path to full dataset (parquet)
        output_file: Path to save toy dataset
        num_samples: Number of samples to include
    """
    print(f"\n{'='*60}")
    print(f"Creating Toy Dataset")
    print(f"{'='*60}\n")

    # Check input file exists
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        print(f"\nPlease download the dataset first:")
        print(f"  hf download dongguanting/ARPO-RL-Reasoning-10K train_10k.parquet \\")
        print(f"    --repo-type dataset --local-dir ./temp_rl")
        print(f"  mv temp_rl/train_10k.parquet rl_datasets/train.parquet")
        sys.exit(1)

    print(f"Reading dataset from: {input_file}")
    df = pd.read_parquet(input_file)

    print(f"Original dataset size: {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")

    # Take first N samples
    df_toy = df.head(num_samples)

    print(f"\nToy dataset size: {len(df_toy)} samples")

    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to parquet
    df_toy.to_parquet(output_file, index=False)

    print(f"✅ Toy dataset saved to: {output_file}")

    # Display sample
    print(f"\n{'='*60}")
    print(f"Sample Entry")
    print(f"{'='*60}\n")

    sample = df_toy.iloc[0]
    print(f"Prompt:")
    print(f"{sample['prompt'][:300]}...")
    print(f"\nAnswer:")
    print(f"{sample['answer']}")
    print(f"\nData Source:")
    print(f"{sample['data_source']}")
    if 'tools' in sample:
        print(f"\nTools:")
        print(f"{sample['tools']}")

    print(f"\n{'='*60}")
    print(f"✅ Toy dataset creation complete!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create toy dataset for local RL testing"
    )
    parser.add_argument(
        '--input',
        default='rl_datasets/train.parquet',
        help='Input parquet file (default: rl_datasets/train.parquet)'
    )
    parser.add_argument(
        '--output',
        default='rl_datasets/train_toy.parquet',
        help='Output parquet file (default: rl_datasets/train_toy.parquet)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='Number of samples to include (default: 100)'
    )

    args = parser.parse_args()

    create_toy_dataset(
        input_file=args.input,
        output_file=args.output,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()
