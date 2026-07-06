"""
Speaker Identification CLI
==========================
Identify a speaker from an audio file using a trained model.

Usage:
    python scripts/identify.py --audio path/to/audio.wav --checkpoint checkpoints/best_model.pt
    python scripts/identify.py --audio path/to/audio.wav --checkpoint checkpoints/best_model.pt --top-k 3
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.pipeline import SpeakerIdentificationPipeline


def main():
    parser = argparse.ArgumentParser(description="Identify Speaker from Audio")
    parser.add_argument(
        "--audio", type=str, required=True,
        help="Path to the audio file to identify",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config", type=str, default="config/inference_config.yaml",
        help="Path to inference configuration",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top predictions to show",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device for inference",
    )
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Initialize pipeline
    print("Loading model...")
    pipeline = SpeakerIdentificationPipeline(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    
    # Run identification
    print(f"\nIdentifying speaker from: {args.audio}")
    print("-" * 40)
    
    results, latency = pipeline.identify(args.audio, top_k=args.top_k)
    
    print(f"\nTop-{args.top_k} Predictions:")
    for i, result in enumerate(results, 1):
        confidence_pct = result["confidence"] * 100
        bar = "#" * int(confidence_pct / 2) + "-" * (50 - int(confidence_pct / 2))
        print(f"  {i}. {result['speaker_id']}: {confidence_pct:.1f}% [{bar}]")
    
    print(f"\nInference time: {latency:.1f} ms")


if __name__ == "__main__":
    main()
