"""
Feature Extractor Module
========================
Extracts acoustic features (MFCC, Mel Spectrogram) from audio waveforms
for speaker embedding networks.
"""

import torch
import torchaudio


class FeatureExtractor:
    """
    Extracts acoustic features from audio waveforms.
    
    Supports MFCC and Mel Spectrogram extraction using torchaudio transforms.
    Features are computed on-the-fly during training and inference.
    
    Args:
        feature_type (str): Type of features to extract ("mfcc" or "mel_spectrogram").
        sample_rate (int): Audio sample rate in Hz.
        n_mfcc (int): Number of MFCC coefficients.
        n_mels (int): Number of Mel filterbank channels.
        n_fft (int): FFT window size.
        win_length (int): Window length in samples.
        hop_length (int): Hop length in samples.
        f_min (float): Minimum frequency for Mel filterbank.
        f_max (float): Maximum frequency for Mel filterbank.
    """
    
    def __init__(
        self,
        feature_type: str = "mfcc",
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        n_mels: int = 80,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        f_min: float = 20.0,
        f_max: float = 7600.0,
    ):
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        
        # Build transforms
        self._build_transforms()
    
    def _build_transforms(self):
        """Initialize the feature extraction transforms."""
        mel_kwargs = {
            "n_mels": self.n_mels,
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "hop_length": self.hop_length,
            "f_min": self.f_min,
            "f_max": self.f_max,
        }
        
        if self.feature_type == "mfcc":
            self.transform = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=self.n_mfcc,
                melkwargs=mel_kwargs,
            )
            self.output_dim = self.n_mfcc
        elif self.feature_type == "mel_spectrogram":
            self.transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                **mel_kwargs,
            )
            self.output_dim = self.n_mels
        else:
            raise ValueError(
                f"Unknown feature type: {self.feature_type}. "
                f"Supported types: 'mfcc', 'mel_spectrogram'"
            )
    
    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract features from an audio waveform.
        
        Args:
            waveform (torch.Tensor): Audio waveform of shape (num_samples,) or
                                      (1, num_samples).
            
        Returns:
            torch.Tensor: Extracted features of shape (n_features, n_frames).
                          For MFCC: (n_mfcc, n_frames)
                          For Mel Spectrogram: (n_mels, n_frames)
        """
        # Ensure 2D input: (1, num_samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Extract features
        features = self.transform(waveform)
        
        # Remove batch dimension → (n_features, n_frames)
        features = features.squeeze(0)
        
        # Apply log scaling for mel spectrogram
        if self.feature_type == "mel_spectrogram":
            features = torch.log(features + 1e-8)
        
        return features
    
    def extract_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a batch of waveforms.
        
        Args:
            waveforms (torch.Tensor): Batch of waveforms of shape (batch, num_samples).
            
        Returns:
            torch.Tensor: Extracted features of shape (batch, n_features, n_frames).
        """
        features = self.transform(waveforms)
        
        if self.feature_type == "mel_spectrogram":
            features = torch.log(features + 1e-8)
        
        return features
    
    @classmethod
    def from_config(cls, config: dict) -> "FeatureExtractor":
        """
        Create a FeatureExtractor from a configuration dictionary.
        
        Args:
            config (dict): Configuration dictionary with feature parameters.
            
        Returns:
            FeatureExtractor: Configured feature extractor instance.
        """
        return cls(
            feature_type=config.get("type", "mfcc"),
            sample_rate=config.get("sample_rate", 16000),
            n_mfcc=config.get("n_mfcc", 40),
            n_mels=config.get("n_mels", 80),
            n_fft=config.get("n_fft", 512),
            win_length=config.get("win_length", 400),
            hop_length=config.get("hop_length", 160),
            f_min=config.get("f_min", 20.0),
            f_max=config.get("f_max", 7600.0),
        )
