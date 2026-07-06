"""
DataLoader Module
=================
Custom DataLoader creation with collation for variable-length features
and balanced sampling across speakers.
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from typing import Dict, List


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collation function for speaker data batches.
    
    Handles variable-length features by padding to the maximum length
    in the batch.
    
    Args:
        batch (list): List of sample dictionaries from SpeakerDataset.
        
    Returns:
        dict: {
            "features": Tensor of shape (batch_size, n_features, max_frames),
            "labels": Tensor of shape (batch_size,),
            "lengths": Tensor of shape (batch_size,) with original frame counts,
        }
    """
    features_list = [item["features"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    
    # Find max frames in this batch
    max_frames = max(f.shape[-1] for f in features_list)
    n_features = features_list[0].shape[0]
    
    # Pad features to max_frames
    padded_features = torch.zeros(len(batch), n_features, max_frames)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    
    for i, feat in enumerate(features_list):
        length = feat.shape[-1]
        padded_features[i, :, :length] = feat
        lengths[i] = length
    
    return {
        "features": padded_features,
        "labels": labels,
        "lengths": lengths,
    }


def create_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    balanced_sampling: bool = False,
    drop_last: bool = True,
) -> DataLoader:
    """
    Create a DataLoader with optional balanced sampling.
    
    Balanced sampling ensures each batch has approximately equal
    representation from each speaker, preventing bias toward
    speakers with more utterances.
    
    Args:
        dataset: SpeakerDataset instance.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle (ignored if balanced_sampling=True).
        num_workers (int): Number of data loading workers.
        pin_memory (bool): Pin memory for GPU transfer.
        balanced_sampling (bool): Use weighted random sampling for class balance.
        drop_last (bool): Drop the last incomplete batch.
        
    Returns:
        DataLoader: Configured DataLoader instance.
    """
    sampler = None
    
    if balanced_sampling:
        # Compute sample weights for balanced sampling
        labels = [item[1] if isinstance(item, tuple) else dataset.speaker_to_label[item[1]]
                  for item in dataset.files]
        label_counts = Counter(labels)
        
        # Weight is inverse of class frequency
        weights = [1.0 / label_counts[label] for label in labels]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        shuffle = False  # Sampler handles shuffling
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=drop_last,
        sampler=sampler,
    )
