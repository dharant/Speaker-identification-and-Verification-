"""
Evaluation Script
=================
Entry point for evaluating the speaker identification model.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --config config/train_config.yaml
"""

import os
import sys
import argparse
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.preprocessing.audio_processor import AudioProcessor
from src.preprocessing.feature_extractor import FeatureExtractor
from src.data.dataset import SpeakerDataset, create_splits
from src.data.dataloader import create_dataloader
from src.models.ecapa_tdnn import ECAPATDNN
from src.models.classifier import ArcFaceClassifier
from src.evaluation.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate Speaker Identification Model")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config", type=str, default="config/train_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save evaluation results and plots",
    )
    parser.add_argument(
        "--num-trials", type=int, default=5000,
        help="Number of verification trials",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device for inference",
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("\n" + "=" * 60)
    print("Speaker Identification System - Evaluation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load model from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_config = checkpoint.get("config", config)
    model_config = ckpt_config.get("model", config.get("model", {}))
    
    model = ECAPATDNN.from_config(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Load classifier
    classifier = None
    if "classifier_state_dict" in checkpoint:
        classifier_state = checkpoint["classifier_state_dict"]
        if "weight" in classifier_state:
            num_classes = classifier_state["weight"].shape[0]
        elif "fc.weight" in classifier_state:
            num_classes = classifier_state["fc.weight"].shape[0]
        else:
            num_classes = 10
        
        classifier = ArcFaceClassifier(
            embedding_dim=model_config.get("embedding_dim", 192),
            num_classes=num_classes,
        ).to(device)
        
        try:
            classifier.load_state_dict(classifier_state)
        except RuntimeError:
            from src.models.classifier import SoftmaxClassifier
            classifier = SoftmaxClassifier(
                embedding_dim=model_config.get("embedding_dim", 192),
                num_classes=num_classes,
            ).to(device)
            classifier.load_state_dict(classifier_state)
        
        classifier.eval()
    
    # Prepare test dataset
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
    )
    
    feature_extractor = FeatureExtractor.from_config(feature_config)
    
    root_dir = dataset_config.get("root_dir", "data/mini_dataset")
    seed = config.get("seed", 42)
    
    _, _, test_files = create_splits(
        root_dir,
        dataset_config.get("train_split", 0.8),
        dataset_config.get("val_split", 0.1),
        dataset_config.get("test_split", 0.1),
        seed,
    )
    
    test_dataset = SpeakerDataset(
        root_dir=root_dir,
        audio_processor=audio_processor,
        feature_extractor=feature_extractor,
        augmentor=None,
        split="test",
        file_list=test_files,
    )
    
    test_loader = create_dataloader(
        test_dataset,
        batch_size=config.get("training", {}).get("batch_size", 32),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    
    print(f"\nTest samples: {len(test_dataset)}")
    print(f"Number of speakers: {test_dataset.get_num_speakers()}")
    
    # Run evaluation
    evaluator = Evaluator(model, device)
    results = evaluator.full_evaluation(
        test_loader,
        classifier=classifier,
        num_trials=args.num_trials,
    )
    
    # Save plots
    print(f"\nSaving plots to {args.output_dir}/...")
    evaluator.save_plots(results, args.output_dir)
    
    # Save evaluation report
    report_path = os.path.join(args.output_dir, "evaluation_results.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("Speaker Identification System - Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Test samples: {results['num_samples']}\n")
        f.write(f"Number of speakers: {results['num_speakers']}\n\n")
        
        f.write("Identification Metrics:\n")
        f.write(f"  Accuracy (Top-1): {results['identification']['identification_accuracy']:.2f}%\n")
        if 'identification_accuracy_top5' in results['identification']:
            f.write(f"  Accuracy (Top-5): {results['identification']['identification_accuracy_top5']:.2f}%\n")
        f.write("\n")
        
        f.write("Verification Metrics:\n")
        f.write(f"  EER: {results['verification']['eer']:.4f} ({results['verification']['eer']*100:.2f}%)\n")
        f.write(f"  EER Threshold: {results['verification']['eer_threshold']:.4f}\n")
        f.write(f"  minDCF: {results['verification']['min_dcf']:.4f}\n")
        f.write(f"  ROC AUC: {results['verification']['roc_auc']:.4f}\n\n")
        
        f.write("Inference Latency:\n")
        f.write(f"  Mean: {results['latency']['mean_latency_ms']:.2f} ms\n")
        f.write(f"  Std: {results['latency']['std_latency_ms']:.2f} ms\n")
        f.write(f"  P95: {results['latency']['p95_latency_ms']:.2f} ms\n")
    
    print(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
