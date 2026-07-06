"""
Inference Pipeline Module
=========================
End-to-end inference pipelines for speaker identification and verification.
Handles raw audio → preprocessing → feature extraction → embedding → prediction.
"""

import os
import sys

# Patch SpeechBrain's LazyModule to support Windows file paths during inspect.py check
try:
    import speechbrain.utils.importutils as sb_import
    original_ensure_module = sb_import.LazyModule.ensure_module
    def patched_ensure_module(self, stacklevel=1):
        try:
            frame = sys._getframe(stacklevel + 1)
            if frame.f_code.co_filename.replace('\\', '/').endswith('/inspect.py'):
                raise AttributeError()
        except AttributeError:
            raise
        except Exception:
            pass
        return original_ensure_module(self, stacklevel)
    sb_import.LazyModule.ensure_module = patched_ensure_module
except Exception:
    pass

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml

from ..preprocessing.audio_processor import AudioProcessor
from ..preprocessing.feature_extractor import FeatureExtractor
from ..models.ecapa_tdnn import ECAPATDNN
from ..models.classifier import ArcFaceClassifier, SoftmaxClassifier
from ..models.verification import SpeakerVerifier


class SpeakerIdentificationPipeline:
    """
    End-to-end speaker identification pipeline.
    
    Takes a raw audio file and returns the predicted speaker identity.
    
    Args:
        config_path (str): Path to inference config YAML.
        checkpoint_path (str): Path to model checkpoint (overrides config).
        device (str): Device for inference ("auto", "cuda", "cpu").
    """
    
    def __init__(
        self,
        config_path: str = "config/inference_config.yaml",
        checkpoint_path: str = None,
        device: str = "auto",
    ):
        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize preprocessing
        preprocess_config = self.config.get("preprocessing", {})
        self.audio_processor = AudioProcessor(
            sample_rate=preprocess_config.get("sample_rate", 16000),
            normalize=preprocess_config.get("normalize", True),
            trim_silence=preprocess_config.get("trim_silence", True),
            trim_db=preprocess_config.get("trim_db", 30),
        )
        
        # Initialize feature extractor
        feature_config = self.config.get("features", {})
        feature_config["sample_rate"] = preprocess_config.get("sample_rate", 16000)
        self.feature_extractor = FeatureExtractor.from_config(feature_config)
        
        # Load model
        ckpt_path = checkpoint_path or self.config.get("model", {}).get("checkpoint_path")
        self._load_model(ckpt_path)
        
        # Speaker label mapping (loaded from checkpoint)
        self.label_to_speaker = {}
    
    def _load_model(self, checkpoint_path: str) -> None:
        """Load model and classifier from checkpoint."""
        self.is_pretrained = False
        self.model_type = "custom"
        
        if checkpoint_path in ["pretrained", "speechbrain", "pretrained_ecapa"]:
            print("Loading SpeechBrain pretrained ECAPA-TDNN model...")
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy
            
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                local_strategy=LocalStrategy.COPY,
                run_opts={"device": str(self.device)}
            )
            self.classifier = None
            self.is_pretrained = True
            self.model_type = "pretrained_ecapa"
            print("Pretrained ECAPA model loaded successfully.")
        elif checkpoint_path == "pretrained_resnet":
            print("Loading SpeechBrain pretrained ResNet-34 model...")
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy
            
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-resnet-voxceleb",
                savedir="pretrained_models/spkrec-resnet-voxceleb",
                local_strategy=LocalStrategy.COPY,
                run_opts={"device": str(self.device)}
            )
            self.classifier = None
            self.is_pretrained = True
            self.model_type = "pretrained_resnet"
            print("Pretrained ResNet model loaded successfully.")
        elif checkpoint_path == "pretrained_wavlm":
            print("Loading Microsoft pretrained WavLM model...")
            from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
            
            self.wavlm_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
            self.model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(self.device)
            self.model.eval()
            self.classifier = None
            self.is_pretrained = True
            self.model_type = "pretrained_wavlm"
            print("Pretrained WavLM model loaded successfully.")
        else:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Get model config from checkpoint
            ckpt_config = checkpoint.get("config", {})
            model_config = ckpt_config.get("model", self.config.get("model", {}))
            
            # Initialize model
            self.model = ECAPATDNN.from_config(model_config).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.is_pretrained = False
            self.model_type = "custom"
            
            # Initialize classifier
            if "classifier_state_dict" in checkpoint:
                classifier_state = checkpoint["classifier_state_dict"]
                # Determine number of classes from the state dict
                if "weight" in classifier_state:
                    num_classes = classifier_state["weight"].shape[0]
                elif "fc.weight" in classifier_state:
                    num_classes = classifier_state["fc.weight"].shape[0]
                else:
                    num_classes = 10  # fallback
                
                embedding_dim = model_config.get("embedding_dim", 192)
                
                # Try ArcFace first, then Softmax
                try:
                    self.classifier = ArcFaceClassifier(
                        embedding_dim=embedding_dim,
                        num_classes=num_classes,
                    ).to(self.device)
                    self.classifier.load_state_dict(classifier_state)
                except RuntimeError:
                    self.classifier = SoftmaxClassifier(
                        embedding_dim=embedding_dim,
                        num_classes=num_classes,
                    ).to(self.device)
                    self.classifier.load_state_dict(classifier_state)
                
                self.classifier.eval()
            else:
                self.classifier = None
            
            print(f"Loaded model from {checkpoint_path}")
    
    @torch.no_grad()
    def identify(
        self, audio_path: str, top_k: int = 5
    ) -> List[Dict[str, float]]:
        """
        Identify the speaker from an audio file.
        
        Args:
            audio_path (str): Path to the audio file.
            top_k (int): Number of top predictions to return.
            
        Returns:
            list: List of {"speaker_id": str, "confidence": float} dicts.
        """
        start_time = time.perf_counter()
        
        # Preprocess audio
        waveform = self.audio_processor.process(audio_path)
        
        # Extract features
        features = self.feature_extractor.extract(waveform)
        features = features.unsqueeze(0).to(self.device)  # Add batch dim
        
        # Get embedding
        embedding = self.model(features, return_embedding=True)
        
        # Classify
        if self.classifier is not None:
            logits = self.classifier(embedding, None)
            probs = F.softmax(logits, dim=1).squeeze()
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probs, min(top_k, probs.shape[0]))
            
            results = []
            for prob, idx in zip(top_probs, top_indices):
                speaker_id = self.label_to_speaker.get(
                    idx.item(), f"speaker_{idx.item():04d}"
                )
                results.append({
                    "speaker_id": speaker_id,
                    "confidence": prob.item(),
                })
        else:
            results = [{"speaker_id": "unknown", "confidence": 0.0}]
        
        elapsed = (time.perf_counter() - start_time) * 1000
        
        return results, elapsed
    
    @torch.no_grad()
    def extract_embedding(self, audio_path: str) -> torch.Tensor:
        """
        Extract speaker embedding from an audio file.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            torch.Tensor: Speaker embedding vector.
        """
        if getattr(self, "is_pretrained", False):
            if self.model_type == "pretrained_wavlm":
                waveform = self.audio_processor.process(audio_path)
                inputs = self.wavlm_extractor(waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                embedding = self.model(**inputs).embeddings.squeeze()
                return embedding.cpu()
            else:
                waveform = self.audio_processor.process(audio_path).unsqueeze(0).to(self.device)
                embedding = self.model.encode_batch(waveform).squeeze()
                return embedding.cpu()
        else:
            waveform = self.audio_processor.process(audio_path)
            features = self.feature_extractor.extract(waveform)
            features = features.unsqueeze(0).to(self.device)
            embedding = self.model(features, return_embedding=True)
            return embedding.squeeze().cpu()


class SpeakerVerificationPipeline:
    """
    End-to-end speaker verification pipeline.
    
    Handles enrollment of speakers and verification of test utterances.
    
    Args:
        config_path (str): Path to inference config YAML.
        checkpoint_path (str): Path to model checkpoint.
        device (str): Device for inference.
    """
    
    def __init__(
        self,
        config_path: str = "config/inference_config.yaml",
        checkpoint_path: str = None,
        device: str = "auto",
    ):
        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize preprocessing
        preprocess_config = self.config.get("preprocessing", {})
        self.audio_processor = AudioProcessor(
            sample_rate=preprocess_config.get("sample_rate", 16000),
            normalize=preprocess_config.get("normalize", True),
            trim_silence=preprocess_config.get("trim_silence", True),
            trim_db=preprocess_config.get("trim_db", 30),
        )
        
        # Initialize feature extractor
        feature_config = self.config.get("features", {})
        feature_config["sample_rate"] = preprocess_config.get("sample_rate", 16000)
        self.feature_extractor = FeatureExtractor.from_config(feature_config)
        
        # Load model
        ckpt_path = checkpoint_path or self.config.get("model", {}).get("checkpoint_path")
        self._load_model(ckpt_path)
        
        # Initialize verifier
        ver_config = self.config.get("verification", {})
        self.verifier = SpeakerVerifier(
            threshold=ver_config.get("threshold", 0.5),
            scoring_method=ver_config.get("scoring_method", "cosine"),
        )
    
    def _load_model(self, checkpoint_path: str) -> None:
        """Load the embedding extraction model."""
        self.is_pretrained = False
        self.model_type = "custom"
        
        if checkpoint_path in ["pretrained", "speechbrain", "pretrained_ecapa"]:
            print("Loading SpeechBrain pretrained ECAPA-TDNN model...")
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy
            
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                local_strategy=LocalStrategy.COPY,
                run_opts={"device": str(self.device)}
            )
            self.is_pretrained = True
            self.model_type = "pretrained_ecapa"
            print("Pretrained ECAPA model loaded successfully.")
        elif checkpoint_path == "pretrained_resnet":
            print("Loading SpeechBrain pretrained ResNet-34 model...")
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy
            
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-resnet-voxceleb",
                savedir="pretrained_models/spkrec-resnet-voxceleb",
                local_strategy=LocalStrategy.COPY,
                run_opts={"device": str(self.device)}
            )
            self.is_pretrained = True
            self.model_type = "pretrained_resnet"
            print("Pretrained ResNet model loaded successfully.")
        elif checkpoint_path == "pretrained_wavlm":
            print("Loading Microsoft pretrained WavLM model...")
            from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
            
            self.wavlm_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
            self.model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(self.device)
            self.model.eval()
            self.is_pretrained = True
            self.model_type = "pretrained_wavlm"
            print("Pretrained WavLM model loaded successfully.")
        else:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            ckpt_config = checkpoint.get("config", {})
            model_config = ckpt_config.get("model", self.config.get("model", {}))
            
            self.model = ECAPATDNN.from_config(model_config).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.is_pretrained = False
            self.model_type = "custom"
            
            print(f"Loaded model from {checkpoint_path}")
    
    @torch.no_grad()
    def _extract_embedding(self, audio_path: str) -> torch.Tensor:
        """Extract embedding from an audio file."""
        if getattr(self, "is_pretrained", False):
            if self.model_type == "pretrained_wavlm":
                waveform = self.audio_processor.process(audio_path)
                inputs = self.wavlm_extractor(waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                embedding = self.model(**inputs).embeddings.squeeze()
                return embedding.cpu()
            else:
                waveform = self.audio_processor.process(audio_path).unsqueeze(0).to(self.device)
                embedding = self.model.encode_batch(waveform).squeeze()
                return embedding.cpu()
        else:
            waveform = self.audio_processor.process(audio_path)
            features = self.feature_extractor.extract(waveform)
            features = features.unsqueeze(0).to(self.device)
            embedding = self.model(features, return_embedding=True)
            return embedding.squeeze().cpu()
    
    def enroll_speaker(
        self, speaker_id: str, audio_paths: List[str]
    ) -> None:
        """
        Enroll a speaker using one or more audio files.
        
        Args:
            speaker_id (str): Unique speaker identifier.
            audio_paths (list): List of enrollment audio file paths.
        """
        embeddings = []
        for path in audio_paths:
            emb = self._extract_embedding(path)
            embeddings.append(emb)
        
        embeddings = torch.stack(embeddings)
        self.verifier.enroll(speaker_id, embeddings, aggregate=True)
        print(f"Enrolled speaker '{speaker_id}' with {len(audio_paths)} utterances")
    
    def verify(
        self, audio_path: str, claimed_speaker_id: str
    ) -> Tuple[bool, float]:
        """
        Verify if an audio file belongs to a claimed speaker.
        
        Args:
            audio_path (str): Path to the test audio file.
            claimed_speaker_id (str): Claimed speaker identity.
            
        Returns:
            Tuple[bool, float]: (accepted, similarity_score).
        """
        start_time = time.perf_counter()
        
        embedding = self._extract_embedding(audio_path)
        accepted, score = self.verifier.verify(embedding, claimed_speaker_id)
        
        elapsed = (time.perf_counter() - start_time) * 1000
        
        return accepted, score, elapsed
    
    def verify_pair(
        self, audio_path1: str, audio_path2: str
    ) -> Tuple[bool, float]:
        """
        Verify if two audio files belong to the same speaker.
        
        Args:
            audio_path1 (str): First audio file path.
            audio_path2 (str): Second audio file path.
            
        Returns:
            Tuple[bool, float]: (same_speaker, similarity_score).
        """
        emb1 = self._extract_embedding(audio_path1)
        emb2 = self._extract_embedding(audio_path2)
        
        score = self.verifier.compute_score(emb1, emb2)
        accepted = score.item() >= self.verifier.threshold
        
        return accepted, score.item()
    
    def get_enrolled_speakers(self) -> List[str]:
        """Return list of enrolled speaker IDs."""
        return self.verifier.get_enrolled_speakers()
