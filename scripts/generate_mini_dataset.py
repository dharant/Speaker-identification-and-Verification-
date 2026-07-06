"""
Mini Dataset Generation Script
===============================
Generate a synthetic mini speaker dataset for testing and demonstration.

Usage:
    python scripts/generate_mini_dataset.py
    python scripts/generate_mini_dataset.py --output data/mini_dataset --speakers 10 --utterances 50
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.mini_dataset import MiniDatasetGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate Mini Speaker Dataset")
    parser.add_argument(
        "--output", type=str, default="data/mini_dataset",
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--speakers", type=int, default=10,
        help="Number of speakers to generate",
    )
    parser.add_argument(
        "--utterances", type=int, default=50,
        help="Number of utterances per speaker",
    )
    parser.add_argument(
        "--duration", type=float, default=3.0,
        help="Duration of each utterance in seconds",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000,
        help="Audio sample rate in Hz",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()
    
    print("=" * 50)
    print("Mini Speaker Dataset Generator")
    print("=" * 50)
    print(f"Output: {args.output}")
    print(f"Speakers: {args.speakers}")
    print(f"Utterances/speaker: {args.utterances}")
    print(f"Duration: {args.duration}s")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"Seed: {args.seed}")
    print()
    
    generator = MiniDatasetGenerator(
        output_dir=args.output,
        num_speakers=args.speakers,
        utterances_per_speaker=args.utterances,
        sample_rate=args.sample_rate,
        duration=args.duration,
        seed=args.seed,
    )
    
    output_path = generator.generate()
    
    print(f"\nDataset ready at: {output_path}")
    print("You can now train using: python scripts/train.py --config config/train_config.yaml")


if __name__ == "__main__":
    main()
