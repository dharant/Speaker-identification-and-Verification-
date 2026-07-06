"""
Evaluator Module
================
Comprehensive evaluation pipeline for speaker identification and
verification systems. Generates detailed reports with metrics, plots,
and analysis.
"""

import os
import time
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .metrics import (
    compute_eer,
    compute_eer_with_threshold,
    compute_dcf,
    compute_accuracy,
    compute_roc_auc,
    compute_confusion_matrix,
)


class Evaluator:
    """
    Comprehensive evaluator for speaker identification/verification systems.
    
    Runs full evaluation suite including:
    - Closed-set speaker identification accuracy
    - Open-set speaker verification (EER, minDCF)
    - ROC curve and AUC analysis
    - Confusion matrix analysis
    - Inference latency measurement
    
    Args:
        model: ECAPA-TDNN model instance.
        device (torch.device): Device for inference.
    """
    
    def __init__(self, model, device: torch.device = None):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def extract_embeddings(self, dataloader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract embeddings for all samples in a DataLoader.
        
        Args:
            dataloader: DataLoader with speaker data.
            
        Returns:
            Tuple[Tensor, Tensor]: (embeddings, labels) tensors.
        """
        all_embeddings = []
        all_labels = []
        
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            features = batch["features"].to(self.device)
            labels = batch["labels"]
            
            embeddings = self.model(features, return_embedding=True)
            
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
        
        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        return embeddings, labels
    
    def evaluate_identification(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        classifier=None,
    ) -> Dict[str, float]:
        """
        Evaluate closed-set speaker identification.
        
        Args:
            embeddings (Tensor): Speaker embeddings (n_samples, embedding_dim).
            labels (Tensor): Ground truth labels (n_samples,).
            classifier: Optional classifier for logit-based identification.
            
        Returns:
            dict: Identification metrics.
        """
        if classifier is not None:
            classifier.eval()
            with torch.no_grad():
                logits = classifier(embeddings.to(self.device), None)
                predictions = logits.argmax(dim=1).cpu()
                predictions_top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices.cpu()
        else:
            # Use nearest-neighbor identification
            predictions = self._nearest_neighbor_identify(embeddings, labels)
            # Use nearest-neighbor top-5
            n = embeddings.shape[0]
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            similarity = torch.mm(embeddings_norm, embeddings_norm.t())
            similarity.fill_diagonal_(-float("inf"))
            nearest_top5 = similarity.topk(min(5, n - 1), dim=1).indices
            predictions_top5 = labels[nearest_top5]
        
        accuracy_top1 = compute_accuracy(
            predictions.numpy(), labels.numpy(), top_k=1
        )
        
        accuracy_top5 = compute_accuracy(
            predictions_top5.numpy(), labels.numpy(), top_k=5
        )
        
        cm = compute_confusion_matrix(
            predictions.numpy(), labels.numpy(),
            num_classes=len(torch.unique(labels)),
        )
        
        return {
            "identification_accuracy": accuracy_top1,
            "identification_accuracy_top5": accuracy_top5,
            "confusion_matrix": cm,
        }
    
    def _nearest_neighbor_identify(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform nearest-neighbor speaker identification.
        
        Uses leave-one-out: for each sample, find the closest embedding
        from a different sample and predict its label.
        """
        n = embeddings.shape[0]
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.mm(embeddings_norm, embeddings_norm.t())
        
        # Zero out self-similarity
        similarity.fill_diagonal_(-float("inf"))
        
        # Predict: label of nearest neighbor
        nearest = similarity.argmax(dim=1)
        predictions = labels[nearest]
        
        return predictions
    
    def evaluate_verification(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        num_trials: int = 5000,
    ) -> Dict[str, float]:
        """
        Evaluate open-set speaker verification.
        
        Generates positive and negative verification trials and
        computes verification metrics.
        
        Args:
            embeddings (Tensor): Speaker embeddings.
            labels (Tensor): Speaker labels.
            num_trials (int): Number of verification trials to generate.
            
        Returns:
            dict: Verification metrics (EER, minDCF, AUC).
        """
        # Generate verification trials
        scores, trial_labels = self._generate_trials(
            embeddings, labels, num_trials
        )
        
        # Compute EER
        eer, eer_threshold = compute_eer_with_threshold(scores, trial_labels)
        
        # Compute minDCF
        min_dcf = compute_dcf(scores, trial_labels)
        
        # Compute ROC/AUC
        fpr, tpr, roc_auc = compute_roc_auc(scores, trial_labels)
        
        return {
            "eer": eer,
            "eer_threshold": eer_threshold,
            "min_dcf": min_dcf,
            "roc_auc": roc_auc,
            "fpr": fpr,
            "tpr": tpr,
            "scores": scores,
            "trial_labels": trial_labels,
        }
    
    def _generate_trials(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        num_trials: int,
    ) -> Tuple[List[float], List[int]]:
        """
        Generate balanced verification trials (same/different speaker pairs).
        
        Args:
            embeddings: Speaker embeddings.
            labels: Speaker labels.
            num_trials: Total number of trials.
            
        Returns:
            Tuple[list, list]: (scores, trial_labels).
        """
        n = embeddings.shape[0]
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Group indices by speaker
        speaker_indices = {}
        for i in range(n):
            label = labels[i].item()
            if label not in speaker_indices:
                speaker_indices[label] = []
            speaker_indices[label].append(i)
        
        scores = []
        trial_labels = []
        speakers = list(speaker_indices.keys())
        
        # Generate equal positive and negative trials
        num_positive = num_trials // 2
        num_negative = num_trials - num_positive
        
        # Positive trials (same speaker)
        for _ in range(num_positive):
            # Pick a speaker with at least 2 utterances
            valid_speakers = [s for s in speakers if len(speaker_indices[s]) >= 2]
            if not valid_speakers:
                break
            
            speaker = random.choice(valid_speakers)
            i, j = random.sample(speaker_indices[speaker], 2)
            
            score = F.cosine_similarity(
                embeddings_norm[i].unsqueeze(0),
                embeddings_norm[j].unsqueeze(0),
            ).item()
            
            scores.append(score)
            trial_labels.append(1)
        
        # Negative trials (different speakers)
        for _ in range(num_negative):
            if len(speakers) < 2:
                break
            
            speaker1, speaker2 = random.sample(speakers, 2)
            i = random.choice(speaker_indices[speaker1])
            j = random.choice(speaker_indices[speaker2])
            
            score = F.cosine_similarity(
                embeddings_norm[i].unsqueeze(0),
                embeddings_norm[j].unsqueeze(0),
            ).item()
            
            scores.append(score)
            trial_labels.append(0)
        
        return scores, trial_labels
    
    def measure_latency(
        self,
        feature_shape: Tuple[int, ...] = (1, 40, 300),
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> Dict[str, float]:
        """
        Measure inference latency.
        
        Args:
            feature_shape: Shape of input features (batch, n_features, n_frames).
            num_runs (int): Number of timing runs.
            warmup_runs (int): Number of warmup runs before timing.
            
        Returns:
            dict: Latency statistics in milliseconds.
        """
        dummy_input = torch.randn(feature_shape).to(self.device)
        
        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                self.model(dummy_input)
        
        # Timing runs
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                self.model(dummy_input)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        latencies = np.array(latencies)
        
        return {
            "mean_latency_ms": float(np.mean(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
        }
    
    def full_evaluation(
        self,
        dataloader,
        classifier=None,
        num_trials: int = 5000,
    ) -> Dict:
        """
        Run the complete evaluation suite.
        
        Args:
            dataloader: Test DataLoader.
            classifier: Optional classifier for identification.
            num_trials: Number of verification trials.
            
        Returns:
            dict: Complete evaluation results.
        """
        print("Running full evaluation...")
        print("=" * 60)
        
        # Extract embeddings
        print("\n1. Extracting embeddings...")
        embeddings, labels = self.extract_embeddings(dataloader)
        print(f"   Extracted {len(embeddings)} embeddings")
        
        # Identification evaluation
        print("\n2. Evaluating speaker identification...")
        id_results = self.evaluate_identification(embeddings, labels, classifier)
        print(f"   Identification Accuracy (Top-1): {id_results['identification_accuracy']:.2f}%")
        if 'identification_accuracy_top5' in id_results:
            print(f"   Identification Accuracy (Top-5): {id_results['identification_accuracy_top5']:.2f}%")
        
        # Verification evaluation
        print("\n3. Evaluating speaker verification...")
        ver_results = self.evaluate_verification(embeddings, labels, num_trials)
        print(f"   EER: {ver_results['eer']:.4f} ({ver_results['eer']*100:.2f}%)")
        print(f"   EER Threshold: {ver_results['eer_threshold']:.4f}")
        print(f"   minDCF: {ver_results['min_dcf']:.4f}")
        print(f"   ROC AUC: {ver_results['roc_auc']:.4f}")
        
        # Latency measurement
        print("\n4. Measuring inference latency...")
        n_features = embeddings.shape[1] if embeddings.dim() > 1 else 192
        latency_results = self.measure_latency()
        print(f"   Mean latency: {latency_results['mean_latency_ms']:.2f} ms")
        print(f"   P95 latency: {latency_results['p95_latency_ms']:.2f} ms")
        
        print("\n" + "=" * 60)
        print("Evaluation complete!")
        
        return {
            "identification": id_results,
            "verification": ver_results,
            "latency": latency_results,
            "num_samples": len(embeddings),
            "num_speakers": len(torch.unique(labels)),
        }
    
    def save_plots(self, results: Dict, output_dir: str = "results") -> None:
        """
        Save evaluation plots (ROC curve, confusion matrix).
        
        Args:
            results (dict): Evaluation results from full_evaluation.
            output_dir (str): Directory to save plots.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("Warning: matplotlib/seaborn not available. Skipping plots.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # ROC Curve
        if "verification" in results:
            ver = results["verification"]
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.plot(ver["fpr"], ver["tpr"], "b-", linewidth=2,
                    label=f'ROC (AUC = {ver["roc_auc"]:.4f})')
            ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random")
            ax.set_xlabel("False Positive Rate", fontsize=12)
            ax.set_ylabel("True Positive Rate", fontsize=12)
            ax.set_title("Speaker Verification ROC Curve", fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
            plt.close(fig)
            print(f"  Saved ROC curve to {output_dir}/roc_curve.png")
        
        # Confusion Matrix
        if "identification" in results:
            cm = results["identification"]["confusion_matrix"]
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted Speaker", fontsize=12)
            ax.set_ylabel("True Speaker", fontsize=12)
            ax.set_title("Speaker Identification Confusion Matrix", fontsize=14)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
            plt.close(fig)
            print(f"  Saved confusion matrix to {output_dir}/confusion_matrix.png")
        
        # Score Distribution
        if "verification" in results:
            ver = results["verification"]
            scores = np.array(ver["scores"])
            trial_labels = np.array(ver["trial_labels"])
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.hist(scores[trial_labels == 1], bins=50, alpha=0.6,
                    label="Same Speaker", color="green", density=True)
            ax.hist(scores[trial_labels == 0], bins=50, alpha=0.6,
                    label="Different Speaker", color="red", density=True)
            ax.axvline(x=ver["eer_threshold"], color="blue", linestyle="--",
                       label=f'EER Threshold ({ver["eer_threshold"]:.3f})')
            ax.set_xlabel("Cosine Similarity Score", fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            ax.set_title("Verification Score Distribution", fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "score_distribution.png"), dpi=150)
            plt.close(fig)
            print(f"  Saved score distribution to {output_dir}/score_distribution.png")
