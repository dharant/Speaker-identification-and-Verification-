"""
Training Script
===============
Entry point for training the speaker identification model.

Usage:
    python scripts/train.py --config config/train_config.yaml
    python scripts/train.py --config config/train_config.yaml --resume checkpoints/checkpoint_epoch_10.pt
    python scripts/train.py --config config/train_config.yaml --epochs 20
"""

import os
import sys
import argparse
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.preprocessing.audio_processor import AudioProcessor
from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.augmentation import AudioAugmentor
from src.data.dataset import SpeakerDataset, create_splits
from src.data.dataloader import create_dataloader
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Speaker Identification Model")
    parser.add_argument(
        "--config", type=str, default="config/train_config.yaml",
        help="Path to training configuration YAML file",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to train on (overrides config)",
    )
    parser.add_argument(
        "--generate-dataset", action="store_true",
        help="Generate mini dataset before training",
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help="Early stopping patience (overrides config)",
    )
    parser.add_argument(
        "--early-stopping", type=str, choices=["true", "false"], default=None,
        help="Enable ('true') or disable ('false') early stopping (overrides config)",
    )
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override config with command-line arguments
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.device is not None:
        config["device"] = args.device
    if args.resume is not None:
        config["checkpoint"]["resume"] = args.resume
    
    # Ensure early_stopping dict exists in config
    if "early_stopping" not in config["training"]:
        config["training"]["early_stopping"] = {}
    if args.patience is not None:
        config["training"]["early_stopping"]["patience"] = args.patience
    if args.early_stopping is not None:
        config["training"]["early_stopping"]["enabled"] = (args.early_stopping == "true")
    
    # Set seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Generate mini dataset if requested
    if args.generate_dataset:
        from src.data.mini_dataset import MiniDatasetGenerator
        dataset_config = config.get("dataset", {})
        generator = MiniDatasetGenerator(
            output_dir=dataset_config.get("root_dir", "data/mini_dataset"),
            num_speakers=dataset_config.get("num_speakers", 10),
            utterances_per_speaker=dataset_config.get("utterances_per_speaker", 50),
            sample_rate=dataset_config.get("sample_rate", 16000),
            duration=dataset_config.get("max_duration", 3.0),
            seed=seed,
        )
        generator.generate()
    
    print("\n" + "=" * 60)
    print("Speaker Identification System - Training")
    print("=" * 60)
    
    # Initialize preprocessing components
    preprocess_config = config.get("preprocessing", {})
    dataset_config = config.get("dataset", {})
    feature_config = config.get("features", {})
    feature_config["sample_rate"] = preprocess_config.get("sample_rate", 16000)
    
    audio_processor = AudioProcessor(
        sample_rate=preprocess_config.get("sample_rate", 16000),
        normalize=preprocess_config.get("normalize", True),
        trim_silence=preprocess_config.get("trim_silence", True),
        trim_db=preprocess_config.get("trim_db", 30),
        max_duration=dataset_config.get("max_duration", 3.0),
        min_duration=dataset_config.get("min_duration", 1.0),
    )
    
    feature_extractor = FeatureExtractor.from_config(feature_config)
    augmentor = AudioAugmentor(config.get("augmentation", {}))
    
    # Create data splits
    root_dir = dataset_config.get("root_dir", "data/mini_dataset")
    print(f"\nDataset: {root_dir}")
    
    train_split = dataset_config.get("train_split", 0.8)
    val_split = dataset_config.get("val_split", 0.1)
    test_split = dataset_config.get("test_split", 0.1)
    
    train_files, val_files, test_files = create_splits(
        root_dir, train_split, val_split, test_split, seed
    )
    
    print(f"Train samples: {len(train_files)}")
    print(f"Val samples: {len(val_files)}")
    print(f"Test samples: {len(test_files)}")
    
    # Create datasets
    train_dataset = SpeakerDataset(
        root_dir=root_dir,
        audio_processor=audio_processor,
        feature_extractor=feature_extractor,
        augmentor=augmentor,
        split="train",
        file_list=train_files,
    )
    
    val_dataset = SpeakerDataset(
        root_dir=root_dir,
        audio_processor=audio_processor,
        feature_extractor=feature_extractor,
        augmentor=None,  # No augmentation for validation
        split="val",
        file_list=val_files,
    )
    
    num_speakers = train_dataset.get_num_speakers()
    print(f"Number of speakers: {num_speakers}")
    
    # Create dataloaders
    training_config = config.get("training", {})
    batch_size = training_config.get("batch_size", 32)
    num_workers = training_config.get("num_workers", 0)  # 0 for Windows compatibility
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=training_config.get("pin_memory", False),
        balanced_sampling=True,
        drop_last=True,
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=training_config.get("pin_memory", False),
        drop_last=False,
    )
    
    # Initialize trainer
    trainer = Trainer(config)
    trainer.setup(num_speakers)
    
    # Resume from checkpoint if specified
    resume_path = config.get("checkpoint", {}).get("resume")
    if resume_path and os.path.exists(resume_path):
        trainer.load_checkpoint(resume_path)
    
    # Train
    history = trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=config["training"]["epochs"],
    )
    
    print("\nTraining Summary:")
    print(f"  Final Train Loss: {history[-1]['train_loss']:.4f}")
    print(f"  Final Train Acc: {history[-1]['train_accuracy']:.2f}%")
    if "val_eer" in history[-1]:
        print(f"  Final Val EER: {history[-1]['val_eer']:.4f}")
    print(f"  Best EER: {trainer.best_eer:.4f}")


if __name__ == "__main__":
    main()
