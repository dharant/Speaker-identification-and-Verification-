"""
Trainer Module
==============
Training loop for speaker identification models with support for:
- Mixed precision training
- Learning rate scheduling with warmup
- Checkpoint saving/loading
- Validation with EER computation
- Logging to console and TensorBoard
"""

import os
import time
import yaml
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from ..models.ecapa_tdnn import ECAPATDNN
from ..models.classifier import ArcFaceClassifier, SoftmaxClassifier
from ..training.losses import ArcFaceLoss, CrossEntropyLoss
from ..evaluation.metrics import compute_eer, compute_accuracy


class Trainer:
    """
    Trainer for speaker identification models.
    
    Manages the complete training lifecycle including model initialization,
    training loop, validation, checkpointing, and logging.
    
    Args:
        config (dict): Training configuration dictionary.
        device (torch.device): Device to train on.
    """
    
    def __init__(self, config: dict, device: torch.device = None):
        self.config = config
        
        # Set device
        if device is not None:
            self.device = device
        elif config.get("device", "auto") == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config["device"])
        
        print(f"Using device: {self.device}")
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_eer = float("inf")
        self.training_history = []
        
        # Initialize components
        self.model = None
        self.classifier = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.criterion = None
        
        # Early stopping configuration
        training_config = config.get("training", {})
        early_stopping_config = training_config.get("early_stopping", {})
        self.early_stopping_enabled = early_stopping_config.get("enabled", False)
        self.early_stopping_patience = early_stopping_config.get("patience", 10)
        self.early_stopping_min_delta = early_stopping_config.get("min_delta", 0.0001)
        self.early_stopping_monitor = early_stopping_config.get("monitor", "val_eer")
        self.early_stopping_counter = 0
        self.early_stopping_best_score = None
    
    def setup(self, num_speakers: int) -> None:
        """
        Initialize model, optimizer, scheduler, and loss function.
        
        Args:
            num_speakers (int): Number of speaker classes.
        """
        model_config = self.config.get("model", {})
        classifier_config = self.config.get("classifier", {})
        training_config = self.config.get("training", {})
        
        # Initialize ECAPA-TDNN model
        self.model = ECAPATDNN.from_config(model_config).to(self.device)
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        # Initialize classifier head
        classifier_type = classifier_config.get("type", "arcface")
        embedding_dim = model_config.get("embedding_dim", 192)
        
        if classifier_type == "arcface":
            arcface_config = classifier_config.get("arcface", {})
            self.classifier = ArcFaceClassifier(
                embedding_dim=embedding_dim,
                num_classes=num_speakers,
                margin=arcface_config.get("margin", 0.2),
                scale=arcface_config.get("scale", 30.0),
            ).to(self.device)
            self.criterion = ArcFaceLoss(
                num_classes=num_speakers,
                embedding_dim=embedding_dim,
            )
        else:
            self.classifier = SoftmaxClassifier(
                embedding_dim=embedding_dim,
                num_classes=num_speakers,
            ).to(self.device)
            self.criterion = CrossEntropyLoss()
        
        # Initialize optimizer
        all_params = list(self.model.parameters()) + list(self.classifier.parameters())
        optimizer_name = training_config.get("optimizer", "adam").lower()
        lr = training_config.get("learning_rate", 0.001)
        weight_decay = training_config.get("weight_decay", 0.0001)
        
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(
                all_params, lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                all_params, lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                all_params, lr=lr, weight_decay=weight_decay,
                momentum=0.9, nesterov=True,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Initialize learning rate scheduler
        scheduler_config = training_config.get("scheduler", {})
        scheduler_type = scheduler_config.get("type", "cosine")
        epochs = training_config.get("epochs", 50)
        warmup_epochs = scheduler_config.get("warmup_epochs", 5)
        min_lr = scheduler_config.get("min_lr", 1e-6)
        
        if scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr
            )
        elif scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )
        elif scheduler_type == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5
            )
        
        # Mixed precision scaler
        use_amp = training_config.get("mixed_precision", True) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=use_amp)
        self.use_amp = use_amp
        
        # Create checkpoint directory
        checkpoint_dir = self.config.get("checkpoint", {}).get("save_dir", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """
        Run one training epoch.
        
        Args:
            dataloader: Training DataLoader.
            
        Returns:
            dict: Training metrics (loss, accuracy).
        """
        self.model.train()
        self.classifier.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        training_config = self.config.get("training", {})
        gradient_clip = training_config.get("gradient_clip", 5.0)
        log_every = self.config.get("logging", {}).get("log_every", 10)
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch + 1}", leave=True)
        
        for batch_idx, batch in enumerate(pbar):
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            with autocast(self.device.type, enabled=self.use_amp):
                embeddings = self.model(features, return_embedding=False)
                logits = self.classifier(embeddings, labels)
                loss = self.criterion(logits, labels)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.classifier.parameters()),
                    gradient_clip,
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = logits.max(dim=1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            self.global_step += 1
            
            # Update progress bar
            if (batch_idx + 1) % log_every == 0 or batch_idx == 0:
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = 100.0 * correct / total
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "acc": f"{accuracy:.2f}%",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                })
        
        metrics = {
            "train_loss": total_loss / len(dataloader),
            "train_accuracy": 100.0 * correct / total,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self, dataloader) -> Dict[str, float]:
        """
        Run validation and compute metrics.
        
        Args:
            dataloader: Validation DataLoader.
            
        Returns:
            dict: Validation metrics (loss, accuracy, EER).
        """
        self.model.eval()
        self.classifier.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_scores = []
        all_labels = []
        all_embeddings = []
        all_speaker_labels = []
        
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Get embeddings and logits
            embeddings = self.model(features, return_embedding=True)
            logits = self.classifier(embeddings, None)  # No margin during validation
            loss = nn.CrossEntropyLoss()(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(dim=1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            all_embeddings.append(embeddings.cpu())
            all_speaker_labels.append(labels.cpu())
        
        # Compute EER from embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_speaker_labels = torch.cat(all_speaker_labels, dim=0)
        
        eer = self._compute_eer_from_embeddings(all_embeddings, all_speaker_labels)
        
        metrics = {
            "val_loss": total_loss / len(dataloader),
            "val_accuracy": 100.0 * correct / max(total, 1),
            "val_eer": eer,
        }
        
        return metrics
    
    def _compute_eer_from_embeddings(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> float:
        """
        Compute EER by generating verification trials from embeddings.
        
        Creates positive (same speaker) and negative (different speaker) pairs
        and computes the Equal Error Rate.
        """
        import torch.nn.functional as F
        import random
        
        n = embeddings.shape[0]
        if n < 4:
            return 0.5  # Not enough samples
        
        # Generate verification pairs
        scores = []
        trial_labels = []
        num_trials = min(1000, n * (n - 1) // 2)
        
        indices = list(range(n))
        
        for _ in range(num_trials):
            i, j = random.sample(indices, 2)
            
            # Cosine similarity
            score = F.cosine_similarity(
                embeddings[i].unsqueeze(0),
                embeddings[j].unsqueeze(0),
            ).item()
            
            # Same speaker = 1, different speaker = 0
            is_same = int(labels[i] == labels[j])
            
            scores.append(score)
            trial_labels.append(is_same)
        
        if len(set(trial_labels)) < 2:
            return 0.5  # Need both positive and negative trials
        
        return compute_eer(scores, trial_labels)
    
    def train(
        self,
        train_dataloader,
        val_dataloader=None,
        epochs: int = None,
    ) -> list:
        """
        Run the complete training loop.
        
        Args:
            train_dataloader: Training DataLoader.
            val_dataloader: Validation DataLoader (optional).
            epochs (int): Number of epochs (overrides config).
            
        Returns:
            list: Training history with metrics per epoch.
        """
        if epochs is None:
            epochs = self.config.get("training", {}).get("epochs", 50)
        
        checkpoint_config = self.config.get("checkpoint", {})
        save_dir = checkpoint_config.get("save_dir", "checkpoints")
        save_every = checkpoint_config.get("save_every", 5)
        save_best = checkpoint_config.get("save_best", True)
        
        scheduler_config = self.config.get("training", {}).get("scheduler", {})
        warmup_epochs = scheduler_config.get("warmup_epochs", 5)
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"{'='*60}")
        
        for epoch in range(self.epoch, epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Warmup learning rate
            if epoch < warmup_epochs:
                warmup_lr = self.config["training"]["learning_rate"] * (epoch + 1) / warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = warmup_lr
            
            # Train
            train_metrics = self.train_epoch(train_dataloader)
            
            # Validate
            val_metrics = {}
            if val_dataloader is not None:
                val_metrics = self.validate(val_dataloader)
            
            # Update scheduler (after warmup)
            if epoch >= warmup_epochs and self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("val_loss", train_metrics["train_loss"]))
                else:
                    self.scheduler.step()
            
            # Compute epoch time
            epoch_time = time.time() - start_time
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics, "epoch_time": epoch_time}
            self.training_history.append(epoch_metrics)
            
            # Print summary
            print(f"\nEpoch {epoch + 1}/{epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f} | "
                  f"Train Acc: {train_metrics['train_accuracy']:.2f}%")
            if val_metrics:
                print(f"  Val Loss: {val_metrics['val_loss']:.4f} | "
                      f"Val Acc: {val_metrics['val_accuracy']:.2f}% | "
                      f"Val EER: {val_metrics['val_eer']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                )
            
            # Save best model
            if save_best and val_metrics:
                current_eer = val_metrics.get("val_eer", float("inf"))
                if current_eer < self.best_eer:
                    self.best_eer = current_eer
                    self.save_checkpoint(
                        os.path.join(save_dir, "best_model.pt")
                    )
                    print(f"  [*] New best EER: {self.best_eer:.4f}")
            
            # Early stopping check
            if self.early_stopping_enabled and val_metrics:
                score = val_metrics.get(self.early_stopping_monitor)
                if score is not None:
                    if self.early_stopping_best_score is None:
                        self.early_stopping_best_score = score
                        self.early_stopping_counter = 0
                    elif score < self.early_stopping_best_score - self.early_stopping_min_delta:
                        self.early_stopping_best_score = score
                        self.early_stopping_counter = 0
                        print(f"  [*] Metric {self.early_stopping_monitor} improved. Resetting early stopping counter.")
                    else:
                        self.early_stopping_counter += 1
                        print(f"  [*] Metric {self.early_stopping_monitor} did not improve. Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                        if self.early_stopping_counter >= self.early_stopping_patience:
                            print(f"\n[!] Early stopping triggered. Training stopped at epoch {epoch + 1}.")
                            break
        
        # Save final model
        self.save_checkpoint(os.path.join(save_dir, "final_model.pt"))
        print(f"\n{'='*60}")
        print(f"Training complete! Best EER: {self.best_eer:.4f}")
        
        return self.training_history
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save a training checkpoint.
        
        Args:
            path (str): Path to save the checkpoint.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_eer": self.best_eer,
            "model_state_dict": self.model.state_dict(),
            "classifier_state_dict": self.classifier.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "training_history": self.training_history,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> None:
        """
        Load a training checkpoint.
        
        Args:
            path (str): Path to the checkpoint file.
            load_optimizer (bool): Whether to load optimizer state.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.epoch = checkpoint.get("epoch", 0) + 1  # Resume from next epoch
        self.global_step = checkpoint.get("global_step", 0)
        self.best_eer = checkpoint.get("best_eer", float("inf"))
        self.training_history = checkpoint.get("training_history", [])
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
