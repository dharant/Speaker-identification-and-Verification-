"""
Audio Processor Module
======================
Handles raw audio loading, resampling, normalization, and silence trimming.
Supports multiple audio formats including WAV, FLAC, and MP3.
"""

import os
import numpy as np
import torch
import soundfile as sf


class AudioProcessor:
    """
    Processes raw audio files for the speaker identification pipeline.
    
    This class handles:
    - Loading audio from various formats (wav, flac, mp3)
    - Resampling to target sample rate
    - Amplitude normalization
    - Silence trimming using energy-based VAD
    - Duration-based cropping/padding
    
    Args:
        sample_rate (int): Target sample rate in Hz. Default: 16000.
        normalize (bool): Whether to normalize audio amplitude. Default: True.
        trim_silence (bool): Whether to trim leading/trailing silence. Default: True.
        trim_db (float): Threshold in dB below peak for silence detection. Default: 30.
        max_duration (float): Maximum audio duration in seconds. Default: 3.0.
        min_duration (float): Minimum audio duration in seconds. Default: 1.0.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        normalize: bool = True,
        trim_silence: bool = True,
        trim_db: float = 30.0,
        max_duration: float = 3.0,
        min_duration: float = 1.0,
    ):
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.trim_silence = trim_silence
        self.trim_db = trim_db
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.max_samples = int(max_duration * sample_rate)
        self.min_samples = int(min_duration * sample_rate)
    
    def load_audio(self, file_path: str) -> torch.Tensor:
        """
        Load an audio file and return as a 1D tensor.
        
        Args:
            file_path (str): Path to the audio file.
            
        Returns:
            torch.Tensor: 1D audio waveform tensor at target sample rate.
            
        Raises:
            FileNotFoundError: If the audio file does not exist.
            RuntimeError: If the audio file cannot be loaded.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        try:
            data, sr = sf.read(file_path, dtype='float32')
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file {file_path}: {e}")
        
        # Convert to torch tensor
        waveform = torch.from_numpy(data)
        
        # Convert to mono if stereo (soundfile returns (samples, channels))
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=1)
        
        # Resample if necessary
        if sr != self.sample_rate:
            from scipy.signal import resample as scipy_resample
            new_length = int(len(waveform) * self.sample_rate / sr)
            waveform_np = scipy_resample(waveform.numpy(), new_length)
            waveform = torch.from_numpy(waveform_np.astype(np.float32))
        
        return waveform
    
    def normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Normalize audio waveform to have zero mean and unit variance.
        
        Args:
            waveform (torch.Tensor): Input audio waveform.
            
        Returns:
            torch.Tensor: Normalized waveform.
        """
        if waveform.abs().max() > 0:
            waveform = waveform - waveform.mean()
            waveform = waveform / (waveform.abs().max() + 1e-8)
        return waveform
    
    def trim_silence_from_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Remove leading and trailing silence from audio using energy-based detection.
        
        Args:
            waveform (torch.Tensor): Input audio waveform.
            
        Returns:
            torch.Tensor: Trimmed waveform.
        """
        if waveform.numel() == 0:
            return waveform
        
        # Compute energy in short frames
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)     # 10ms hop
        
        # Compute frame energies
        num_frames = max(1, (waveform.shape[0] - frame_length) // hop_length + 1)
        energies = torch.zeros(num_frames)
        
        for i in range(num_frames):
            start = i * hop_length
            end = min(start + frame_length, waveform.shape[0])
            frame = waveform[start:end]
            energies[i] = (frame ** 2).mean()
        
        # Convert to dB
        energies_db = 10 * torch.log10(energies + 1e-10)
        max_energy = energies_db.max()
        threshold = max_energy - self.trim_db
        
        # Find non-silent frames
        active_frames = (energies_db >= threshold).nonzero(as_tuple=True)[0]
        
        if len(active_frames) == 0:
            return waveform
        
        start_frame = active_frames[0].item()
        end_frame = active_frames[-1].item()
        
        start_sample = start_frame * hop_length
        end_sample = min((end_frame + 1) * hop_length + frame_length, waveform.shape[0])
        
        return waveform[start_sample:end_sample]
    
    def fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Pad or crop waveform to the target duration.
        
        Short signals are zero-padded, long signals are randomly cropped.
        
        Args:
            waveform (torch.Tensor): Input audio waveform.
            
        Returns:
            torch.Tensor: Fixed-length waveform.
        """
        current_length = waveform.shape[0]
        
        if current_length >= self.max_samples:
            # Random crop for longer signals
            start = torch.randint(0, current_length - self.max_samples + 1, (1,)).item()
            waveform = waveform[start:start + self.max_samples]
        elif current_length < self.max_samples:
            # Zero-pad for shorter signals
            padding = self.max_samples - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding), value=0.0)
        
        return waveform
    
    def process(self, file_path: str) -> torch.Tensor:
        """
        Full preprocessing pipeline: load → trim → normalize → fix length.
        
        Args:
            file_path (str): Path to the audio file.
            
        Returns:
            torch.Tensor: Preprocessed audio waveform of fixed length.
        """
        # Load audio
        waveform = self.load_audio(file_path)
        
        # Trim silence
        if self.trim_silence:
            waveform = self.trim_silence_from_audio(waveform)
        
        # Normalize
        if self.normalize:
            waveform = self.normalize_audio(waveform)
        
        # Fix length
        waveform = self.fix_length(waveform)
        
        return waveform
    
    def process_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Process an already-loaded waveform (skip file loading step).
        
        Args:
            waveform (torch.Tensor): Input audio waveform.
            
        Returns:
            torch.Tensor: Preprocessed waveform.
        """
        if self.trim_silence:
            waveform = self.trim_silence_from_audio(waveform)
        
        if self.normalize:
            waveform = self.normalize_audio(waveform)
        
        waveform = self.fix_length(waveform)
        
        return waveform
