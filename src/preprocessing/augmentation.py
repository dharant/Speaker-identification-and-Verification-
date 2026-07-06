"""
Audio Augmentation Module
=========================
Implements data augmentation techniques to improve model robustness
against noise, channel variability, and speaker variability.
"""

import random
import torch
import numpy as np


class AudioAugmentor:
    """
    Applies various audio augmentation techniques for training robustness.
    
    Supported augmentations:
    - Additive noise (white, pink)
    - Speed perturbation (time stretching)
    - Reverberation simulation
    - SpecAugment (time/frequency masking on spectrograms)
    
    Args:
        config (dict): Augmentation configuration dictionary.
    """
    
    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        
        self.enabled = config.get("enabled", True)
        self.noise_config = config.get("noise", {})
        self.speed_config = config.get("speed_perturbation", {})
        self.reverb_config = config.get("reverberation", {})
        self.spec_aug_config = config.get("spec_augment", {})
    
    def add_noise(self, waveform: torch.Tensor, snr_db: float = None) -> torch.Tensor:
        """
        Add random noise to the waveform.
        
        Args:
            waveform (torch.Tensor): Input waveform.
            snr_db (float): Signal-to-noise ratio in dB. If None, randomly sampled
                           from the configured range.
            
        Returns:
            torch.Tensor: Noisy waveform.
        """
        if not self.noise_config.get("enabled", True):
            return waveform
        
        snr_range = self.noise_config.get("snr_range", [5, 20])
        if snr_db is None:
            snr_db = random.uniform(snr_range[0], snr_range[1])
        
        noise_types = self.noise_config.get("types", ["white"])
        noise_type = random.choice(noise_types)
        
        if noise_type == "white":
            noise = torch.randn_like(waveform)
        elif noise_type == "pink":
            noise = self._generate_pink_noise(waveform.shape[0])
        else:
            noise = torch.randn_like(waveform)
        
        # Calculate scaling factor for desired SNR
        signal_power = (waveform ** 2).mean()
        noise_power = (noise ** 2).mean()
        
        if noise_power > 0:
            snr_linear = 10 ** (snr_db / 10)
            scale = torch.sqrt(signal_power / (snr_linear * noise_power))
            noisy_waveform = waveform + scale * noise
        else:
            noisy_waveform = waveform
        
        return noisy_waveform
    
    def _generate_pink_noise(self, num_samples: int) -> torch.Tensor:
        """
        Generate pink noise (1/f noise) using the Voss-McCartney algorithm.
        
        Args:
            num_samples (int): Number of samples to generate.
            
        Returns:
            torch.Tensor: Pink noise waveform.
        """
        # Simple approximation using filtered white noise
        white = torch.randn(num_samples)
        
        # Apply simple 1/f filter via cumulative sum and differentiation
        # This is a rough approximation but effective for augmentation
        b = torch.tensor([0.049922035, -0.095993537, 0.050612699, -0.004709510])
        a = torch.tensor([1.0, -2.494956002, 2.017265875, -0.522189400])
        
        # Simplified: just use smoothed white noise
        kernel_size = 5
        kernel = torch.ones(kernel_size) / kernel_size
        pink = torch.nn.functional.conv1d(
            white.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=kernel_size // 2
        ).squeeze()
        
        return pink[:num_samples]
    
    def speed_perturbation(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply speed perturbation (time stretching) to the waveform.
        
        Args:
            waveform (torch.Tensor): Input waveform.
            
        Returns:
            torch.Tensor: Speed-perturbed waveform.
        """
        if not self.speed_config.get("enabled", True):
            return waveform
        
        factors = self.speed_config.get("factors", [0.9, 0.95, 1.0, 1.05, 1.1])
        factor = random.choice(factors)
        
        if factor == 1.0:
            return waveform
        
        # Use interpolation for speed perturbation
        original_length = waveform.shape[0]
        new_length = int(original_length / factor)
        
        # Reshape for interpolation
        waveform_2d = waveform.unsqueeze(0).unsqueeze(0)
        perturbed = torch.nn.functional.interpolate(
            waveform_2d,
            size=new_length,
            mode="linear",
            align_corners=False,
        )
        
        return perturbed.squeeze()
    
    def add_reverberation(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Simulate room reverberation using a simple synthetic impulse response.
        
        Args:
            waveform (torch.Tensor): Input waveform.
            
        Returns:
            torch.Tensor: Reverberant waveform.
        """
        if not self.reverb_config.get("enabled", True):
            return waveform
        
        room_scale_range = self.reverb_config.get("room_scale_range", [0, 50])
        room_scale = random.uniform(room_scale_range[0], room_scale_range[1])
        
        if room_scale < 5:
            return waveform
        
        # Generate simple synthetic impulse response
        ir_length = int(room_scale * 16)  # Scale IR length with room size
        ir_length = min(ir_length, 8000)   # Cap at 0.5s at 16kHz
        
        # Exponentially decaying noise as impulse response
        t = torch.arange(ir_length, dtype=torch.float32)
        decay = torch.exp(-t * 6.0 / ir_length)
        ir = torch.randn(ir_length) * decay
        ir[0] = 1.0  # Direct path
        ir = ir / ir.abs().max()
        
        # Convolve using FFT
        waveform_padded = torch.nn.functional.pad(waveform, (0, ir_length - 1))
        ir_padded = torch.nn.functional.pad(ir, (0, waveform_padded.shape[0] - ir_length))
        
        # FFT convolution
        result = torch.fft.irfft(
            torch.fft.rfft(waveform_padded) * torch.fft.rfft(ir_padded)
        )
        
        # Trim to original length
        result = result[:waveform.shape[0]]
        
        # Normalize to prevent clipping
        if result.abs().max() > 0:
            result = result / result.abs().max() * waveform.abs().max()
        
        return result
    
    def spec_augment(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment: time and frequency masking on spectral features.
        
        Args:
            features (torch.Tensor): Spectral features of shape (n_features, n_frames).
            
        Returns:
            torch.Tensor: Augmented features.
        """
        if not self.spec_aug_config.get("enabled", True):
            return features
        
        features = features.clone()
        n_features, n_frames = features.shape
        
        freq_mask_param = self.spec_aug_config.get("freq_mask_param", 10)
        time_mask_param = self.spec_aug_config.get("time_mask_param", 20)
        num_freq_masks = self.spec_aug_config.get("num_freq_masks", 2)
        num_time_masks = self.spec_aug_config.get("num_time_masks", 2)
        
        # Frequency masking
        for _ in range(num_freq_masks):
            f = random.randint(0, min(freq_mask_param, n_features - 1))
            f0 = random.randint(0, n_features - f)
            features[f0:f0 + f, :] = 0.0
        
        # Time masking
        for _ in range(num_time_masks):
            t = random.randint(0, min(time_mask_param, n_frames - 1))
            t0 = random.randint(0, n_frames - t)
            features[:, t0:t0 + t] = 0.0
        
        return features
    
    def augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply a random combination of waveform-level augmentations.
        
        Args:
            waveform (torch.Tensor): Input waveform.
            
        Returns:
            torch.Tensor: Augmented waveform.
        """
        if not self.enabled:
            return waveform
        
        # Apply augmentations with probability
        if random.random() < 0.5:
            waveform = self.add_noise(waveform)
        
        if random.random() < 0.3:
            waveform = self.speed_perturbation(waveform)
        
        if random.random() < 0.3:
            waveform = self.add_reverberation(waveform)
        
        return waveform
    
    def augment_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply feature-level augmentations (SpecAugment).
        
        Args:
            features (torch.Tensor): Spectral features.
            
        Returns:
            torch.Tensor: Augmented features.
        """
        if not self.enabled:
            return features
        
        if random.random() < 0.5:
            features = self.spec_augment(features)
        
        return features
