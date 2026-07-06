"""
Mini Dataset Generator
======================
Generates a small synthetic speaker dataset for testing and demonstration.
Each speaker has a unique voice signature created by combining different
frequency characteristics.
"""

import os
import math
import random
from pathlib import Path

import torch
import soundfile as sf


class MiniDatasetGenerator:
    """
    Generates a synthetic mini speaker dataset for testing.
    
    Creates distinct speaker profiles using combinations of:
    - Fundamental frequency (pitch)
    - Formant frequencies
    - Speaking rate variations
    - Spectral envelope characteristics
    
    Args:
        output_dir (str): Directory to save generated audio files.
        num_speakers (int): Number of speakers to generate.
        utterances_per_speaker (int): Number of utterances per speaker.
        sample_rate (int): Audio sample rate in Hz.
        duration (float): Duration of each utterance in seconds.
        seed (int): Random seed for reproducibility.
    """
    
    def __init__(
        self,
        output_dir: str = "data/mini_dataset",
        num_speakers: int = 10,
        utterances_per_speaker: int = 50,
        sample_rate: int = 16000,
        duration: float = 3.0,
        seed: int = 42,
    ):
        self.output_dir = output_dir
        self.num_speakers = num_speakers
        self.utterances_per_speaker = utterances_per_speaker
        self.sample_rate = sample_rate
        self.duration = duration
        self.seed = seed
        self.num_samples = int(sample_rate * duration)
    
    def generate(self) -> str:
        """
        Generate the complete mini dataset.
        
        Returns:
            str: Path to the generated dataset directory.
        """
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate unique speaker profiles
        speaker_profiles = self._generate_speaker_profiles()
        
        total_files = 0
        for speaker_idx in range(self.num_speakers):
            speaker_id = f"speaker_{speaker_idx:04d}"
            speaker_dir = output_path / speaker_id
            speaker_dir.mkdir(parents=True, exist_ok=True)
            
            profile = speaker_profiles[speaker_idx]
            
            for utt_idx in range(self.utterances_per_speaker):
                # Generate utterance with speaker characteristics + variation
                waveform = self._synthesize_utterance(profile, utt_idx)
                
                # Save as WAV
                filename = f"utterance_{utt_idx:04d}.wav"
                filepath = speaker_dir / filename
                sf.write(
                    str(filepath),
                    waveform.numpy(),
                    self.sample_rate,
                )
                total_files += 1
        
        print(f"Generated mini dataset:")
        print(f"  Directory: {output_path}")
        print(f"  Speakers: {self.num_speakers}")
        print(f"  Utterances per speaker: {self.utterances_per_speaker}")
        print(f"  Total files: {total_files}")
        print(f"  Duration per utterance: {self.duration}s")
        print(f"  Sample rate: {self.sample_rate} Hz")
        
        return str(output_path)
    
    def _generate_speaker_profiles(self) -> list:
        """
        Generate unique voice profiles for each speaker.
        
        Each profile contains parameters that define a speaker's
        unique voice characteristics.
        
        Returns:
            list: List of speaker profile dictionaries.
        """
        profiles = []
        
        for i in range(self.num_speakers):
            profile = {
                # Fundamental frequency (pitch) - varies between speakers
                "f0": random.uniform(80, 300),
                # Formant frequencies (vocal tract resonances)
                "formants": [
                    random.uniform(300, 900),    # F1
                    random.uniform(900, 2500),   # F2
                    random.uniform(2500, 3500),  # F3
                ],
                # Formant bandwidths
                "bandwidths": [
                    random.uniform(50, 150),
                    random.uniform(70, 200),
                    random.uniform(100, 250),
                ],
                # Speaking rate (affects temporal patterns)
                "speaking_rate": random.uniform(0.8, 1.2),
                # Jitter (pitch variation)
                "jitter": random.uniform(0.01, 0.05),
                # Shimmer (amplitude variation)
                "shimmer": random.uniform(0.02, 0.08),
                # Spectral tilt (voice quality)
                "spectral_tilt": random.uniform(-6, -2),
                # Noise level (breathiness)
                "noise_level": random.uniform(0.01, 0.05),
            }
            profiles.append(profile)
        
        return profiles
    
    def _synthesize_utterance(
        self, profile: dict, utterance_idx: int
    ) -> torch.Tensor:
        """
        Synthesize a single utterance based on a speaker profile.
        
        Creates a voice-like signal using additive synthesis with
        speaker-specific parameters and utterance-level variation.
        
        Args:
            profile (dict): Speaker profile parameters.
            utterance_idx (int): Utterance index (for variation).
            
        Returns:
            torch.Tensor: Synthesized waveform of shape (num_samples,).
        """
        t = torch.arange(self.num_samples, dtype=torch.float32) / self.sample_rate
        
        # Add utterance-level variation to base parameters
        f0 = profile["f0"] * (1 + random.uniform(-0.05, 0.05))
        
        # Generate glottal pulse train (source signal)
        # Fundamental + harmonics
        signal = torch.zeros(self.num_samples)
        num_harmonics = min(20, int(self.sample_rate / 2 / f0))
        
        for h in range(1, num_harmonics + 1):
            freq = f0 * h
            if freq >= self.sample_rate / 2:
                break
            
            # Add jitter to each harmonic
            jitter = 1 + random.uniform(-profile["jitter"], profile["jitter"])
            freq *= jitter
            
            # Amplitude decreases with harmonic number (spectral tilt)
            amplitude = math.pow(h, profile["spectral_tilt"] / 10)
            
            # Add shimmer
            shimmer = 1 + random.uniform(-profile["shimmer"], profile["shimmer"])
            amplitude *= shimmer
            
            # Phase variation per utterance
            phase = random.uniform(0, 2 * math.pi)
            
            signal += amplitude * torch.sin(2 * math.pi * freq * t + phase)
        
        # Apply formant filtering (simplified)
        for i, (formant, bw) in enumerate(
            zip(profile["formants"], profile["bandwidths"])
        ):
            # Formant resonance as amplitude modulation
            formant_varied = formant * (1 + random.uniform(-0.03, 0.03))
            resonance = torch.exp(
                -0.5 * ((t * self.sample_rate % (self.sample_rate / formant_varied)) 
                         / (bw / formant_varied * self.sample_rate)) ** 2
            )
            signal *= (0.5 + 0.5 * resonance)
        
        # Add speech-like amplitude envelope (syllable structure)
        rate = profile["speaking_rate"] * (1 + random.uniform(-0.1, 0.1))
        syllable_rate = random.uniform(3, 6) * rate  # 3-6 syllables/sec
        envelope = 0.5 + 0.5 * torch.sin(2 * math.pi * syllable_rate * t)
        envelope = torch.clamp(envelope, 0.1, 1.0)
        signal *= envelope
        
        # Add breathiness noise
        noise = torch.randn(self.num_samples) * profile["noise_level"]
        signal += noise
        
        # Normalize
        signal = signal / (signal.abs().max() + 1e-8) * 0.9
        
        return signal


def main():
    """Generate a mini dataset with default parameters."""
    generator = MiniDatasetGenerator()
    generator.generate()


if __name__ == "__main__":
    main()
