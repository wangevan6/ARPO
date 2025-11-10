#!/usr/bin/env python3
"""
Script to create a toy dataset subset from the full ARPO-SFT-54K dataset.
This is used for local testing before running full-scale training.

Usage:
    python create_toy_dataset.py [--num_samples 500] [--input_file PATH] [--output_file PATH]
"""

import json
import argparse
from pathlib import Path


def create_toy_dataset(input_file: str, output_file: str, num_samples: int = 500):
    """
    Extract the first N samples from a dataset and save as a toy subset.

    Args:
        input_file: Path to full dataset (JSON or JSONL)
        output_file: Path to save toy dataset
        num_samples: Number of samples to extract (default: 500)
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    # Check if input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading dataset from: {input_file}")

    # Read input file
    if input_path.suffix == '.jsonl':
        # JSONL format (one JSON object per line)
        samples = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                samples.append(json.loads(line))
    else:
        # JSON format (list of objects or single object)
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            samples = data[:num_samples]
        else:
            # If it's a dict, assume it's a single sample
            samples = [data]

    print(f"Extracted {len(samples)} samples from {len(samples)} available")

    # Validate format (check if it's ShareGPT format)
    if samples and 'conversations' in samples[0]:
        print("✓ Detected ShareGPT format")
        print(f"  Sample conversation turns: {len(samples[0]['conversations'])}")
    else:
        print("⚠ Warning: Dataset may not be in ShareGPT format")
        if samples:
            print(f"  Available keys: {list(samples[0].keys())}")

    # Write toy dataset
    output_ext = output_path.suffix

    if output_ext == '.jsonl':
        # Write as JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"✓ Saved {len(samples)} samples to: {output_file} (JSONL format)")
    else:
        # Write as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved {len(samples)} samples to: {output_file} (JSON format)")

    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total samples: {len(samples)}")

    if samples and 'conversations' in samples[0]:
        total_turns = sum(len(s.get('conversations', [])) for s in samples)
        avg_turns = total_turns / len(samples)
        print(f"  Total conversation turns: {total_turns}")
        print(f"  Average turns per sample: {avg_turns:.2f}")

    # Show first sample
    if samples:
        print("\nFirst sample preview:")
        print(json.dumps(samples[0], ensure_ascii=False, indent=2)[:500] + "...")

    return len(samples)


def main():
    parser = argparse.ArgumentParser(
        description="Create a toy dataset subset for local SFT testing"
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=500,
        help='Number of samples to extract (default: 500)'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default='./data/final_sft_edition9.jsonl',
        help='Path to full dataset file'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='./data/final_sft_edition9_toy.jsonl',
        help='Path to save toy dataset'
    )

    args = parser.parse_args()

    try:
        create_toy_dataset(
            input_file=args.input_file,
            output_file=args.output_file,
            num_samples=args.num_samples
        )
        print("\n✓ Toy dataset created successfully!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
