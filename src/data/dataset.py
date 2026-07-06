"""
Speaker Dataset Module
======================
PyTorch Dataset implementation for loading speaker audio data.
Supports VoxCeleb directory structure and the mini dataset format.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from ..preprocessing.audio_processor import AudioProcessor
from ..preprocessing.feature_extractor import FeatureExtractor
from ..preprocessing.augmentation import AudioAugmentor


class SpeakerDataset(Dataset):
    """
    PyTorch Dataset for speaker identification/verification.
    
    Supports directory structures:
    - VoxCeleb: root/speaker_id/session_id/utterance.wav
    - Mini dataset: root/speaker_id/utterance.wav
    
    Args:
        root_dir (str): Root directory containing speaker audio files.
        audio_processor (AudioProcessor): Audio preprocessing instance.
        feature_extractor (FeatureExtractor): Feature extraction instance.
        augmentor (AudioAugmentor): Data augmentation instance (None for no augmentation).
        split (str): Dataset split ("train", "val", "test").
        speaker_ids (list): List of speaker IDs to include (None for all).
        file_list (list): Pre-computed list of (file_path, speaker_label) tuples.
        max_files_per_speaker (int): Maximum files per speaker (None for all).
    """
    
    def __init__(
        self,
        root_dir: str,
        audio_processor: AudioProcessor,
        feature_extractor: FeatureExtractor,
        augmentor: Optional[AudioAugmentor] = None,
        split: str = "train",
        speaker_ids: Optional[List[str]] = None,
        file_list: Optional[List[Tuple[str, int]]] = None,
        max_files_per_speaker: Optional[int] = None,
    ):
        self.root_dir = root_dir
        self.audio_processor = audio_processor
        self.feature_extractor = feature_extractor
        self.augmentor = augmentor if split == "train" else None
        self.split = split
        
        if file_list is not None:
            self.files = file_list
        else:
            self.files = self._scan_directory(speaker_ids, max_files_per_speaker)
        
        # Build speaker label mapping
        self._build_label_map()
    
    def _scan_directory(
        self,
        speaker_ids: Optional[List[str]] = None,
        max_files_per_speaker: Optional[int] = None,
    ) -> List[Tuple[str, str]]:
        """
        Scan the root directory for audio files organized by speaker.
        
        Returns:
            List of (file_path, speaker_id) tuples.
        """
        files = []
        root = Path(self.root_dir)
        
        if not root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
        
        # Get speaker directories
        speaker_dirs = sorted([
            d for d in root.iterdir()
            if d.is_dir() and (speaker_ids is None or d.name in speaker_ids)
        ])
        
        for speaker_dir in speaker_dirs:
            speaker_id = speaker_dir.name
            speaker_files = []
            
            # Recursively find audio files
            for ext in ["*.wav", "*.flac", "*.mp3"]:
                speaker_files.extend(list(speaker_dir.rglob(ext)))
            
            # Sort for reproducibility
            speaker_files.sort()
            
            # Limit files per speaker if specified
            if max_files_per_speaker is not None:
                speaker_files = speaker_files[:max_files_per_speaker]
            
            for f in speaker_files:
                files.append((str(f), speaker_id))
        
        if len(files) == 0:
            raise RuntimeError(
                f"No audio files found in {self.root_dir}. "
                f"Expected directory structure: root/speaker_id/[session/]*.wav"
            )
        
        return files
    
    def _build_label_map(self):
        """Build mapping from speaker IDs to integer labels."""
        unique_speakers = sorted(set(sid for _, sid in self.files))
        self.speaker_to_label = {sid: i for i, sid in enumerate(unique_speakers)}
        self.label_to_speaker = {i: sid for sid, i in self.speaker_to_label.items()}
        self.num_speakers = len(unique_speakers)
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx (int): Sample index.
            
        Returns:
            dict: {
                "features": Tensor of shape (n_features, n_frames),
                "label": int speaker label,
                "speaker_id": str speaker ID,
            }
        """
        file_path, speaker_id = self.files[idx]
        label = self.speaker_to_label[speaker_id]
        
        # Load and preprocess audio
        try:
            waveform = self.audio_processor.process(file_path)
        except Exception as e:
            # Fallback: return a zero tensor if file can't be loaded
            print(f"Warning: Could not load {file_path}: {e}")
            waveform = torch.zeros(
                int(self.audio_processor.max_duration * self.audio_processor.sample_rate)
            )
        
        # Apply waveform augmentation (training only)
        if self.augmentor is not None:
            waveform = self.augmentor.augment_waveform(waveform)
        
        # Extract features
        features = self.feature_extractor.extract(waveform)
        
        # Apply feature augmentation (SpecAugment, training only)
        if self.augmentor is not None:
            features = self.augmentor.augment_features(features)
        
        return {
            "features": features,
            "label": label,
            "speaker_id": speaker_id,
        }
    
    def get_num_speakers(self) -> int:
        """Return the number of unique speakers."""
        return self.num_speakers
    
    def get_speaker_label(self, speaker_id: str) -> int:
        """Get the integer label for a speaker ID."""
        return self.speaker_to_label.get(speaker_id, -1)
    
    def get_speaker_id(self, label: int) -> str:
        """Get the speaker ID for an integer label."""
        return self.label_to_speaker.get(label, "unknown")


def create_splits(
    root_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Split dataset files into train/val/test sets per speaker.
    
    Each speaker's utterances are split proportionally to ensure
    all speakers are represented in all splits.
    
    Args:
        root_dir (str): Root directory of the dataset.
        train_ratio (float): Proportion for training.
        val_ratio (float): Proportion for validation.
        test_ratio (float): Proportion for testing.
        seed (int): Random seed.
        
    Returns:
        Tuple of (train_files, val_files, test_files).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    random.seed(seed)
    root = Path(root_dir)
    
    train_files = []
    val_files = []
    test_files = []
    
    # Process each speaker
    for speaker_dir in sorted(root.iterdir()):
        if not speaker_dir.is_dir():
            continue
        
        speaker_id = speaker_dir.name
        speaker_files = []
        
        for ext in ["*.wav", "*.flac", "*.mp3"]:
            speaker_files.extend(list(speaker_dir.rglob(ext)))
        
        speaker_files.sort()
        random.shuffle(speaker_files)
        
        n = len(speaker_files)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        
        for f in speaker_files[:n_train]:
            train_files.append((str(f), speaker_id))
        for f in speaker_files[n_train:n_train + n_val]:
            val_files.append((str(f), speaker_id))
        for f in speaker_files[n_train + n_val:]:
            test_files.append((str(f), speaker_id))
    
    return train_files, val_files, test_files
