"""
Speaker Verification CLI
========================
Verify if two audio files belong to the same speaker.

Usage:
    python scripts/verify.py --audio1 path/to/file1.wav --audio2 path/to/file2.wav --checkpoint checkpoints/best_model.pt
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

import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from src.preprocessing.audio_processor import AudioProcessor
from src.preprocessing.feature_extractor import FeatureExtractor
from src.models.ecapa_tdnn import ECAPATDNN
from src.models.verification import SpeakerVerifier


def main():
    parser = argparse.ArgumentParser(description="Verify if Two Audio Files are Same Speaker")
    parser.add_argument(
        "--audio1", type=str, required=True,
        help="Path to the first audio file",
    )
    parser.add_argument(
        "--audio2", type=str, required=True,
        help="Path to the second audio file",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config", type=str, default="config/inference_config.yaml",
        help="Path to inference configuration",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Verification threshold (overrides config)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device for inference",
    )
    args = parser.parse_args()
    
    # Validate inputs
    for audio_path in [args.audio1, args.audio2]:
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found: {audio_path}")
            sys.exit(1)
    
    if args.checkpoint not in ["pretrained", "pretrained_ecapa", "pretrained_resnet", "pretrained_wavlm"] and not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Initialize components
    preprocess_config = config.get("preprocessing", {})
    audio_processor = AudioProcessor(
        sample_rate=preprocess_config.get("sample_rate", 16000),
        normalize=preprocess_config.get("normalize", True),
        trim_silence=preprocess_config.get("trim_silence", True),
        trim_db=preprocess_config.get("trim_db", 30),
    )
    
    feature_config = config.get("features", {})
    feature_config["sample_rate"] = preprocess_config.get("sample_rate", 16000)
    feature_extractor = FeatureExtractor.from_config(feature_config)
    
    # Load model
    print("Loading model...")
    is_pretrained = False
    model_type = "custom"
    
    if args.checkpoint in ["pretrained", "pretrained_ecapa"]:
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": str(device)}
        )
        is_pretrained = True
        model_type = "pretrained_ecapa"
        print("Loaded pretrained model speechbrain/spkrec-ecapa-voxceleb")
    elif args.checkpoint == "pretrained_resnet":
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-resnet-voxceleb",
            savedir="pretrained_models/spkrec-resnet-voxceleb",
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": str(device)}
        )
        is_pretrained = True
        model_type = "pretrained_resnet"
        print("Loaded pretrained model speechbrain/spkrec-resnet-voxceleb")
    elif args.checkpoint == "pretrained_wavlm":
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
        
        wavlm_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device)
        model.eval()
        is_pretrained = True
        model_type = "pretrained_wavlm"
        print("Loaded pretrained model microsoft/wavlm-base-plus-sv")
    else:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        ckpt_config = checkpoint.get("config", config)
        model_config = ckpt_config.get("model", config.get("model", {}))
        
        model = ECAPATDNN.from_config(model_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        is_pretrained = False
        model_type = "custom"
        print(f"Loaded model from {args.checkpoint}")
    
    # Set threshold
    ver_config = config.get("verification", {})
    threshold = args.threshold or ver_config.get("threshold", 0.5)
    
    verifier = SpeakerVerifier(
        threshold=threshold,
        scoring_method=ver_config.get("scoring_method", "cosine"),
    )
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    
    with torch.no_grad():
        if is_pretrained:
            if model_type == "pretrained_wavlm":
                # Audio 1
                waveform1 = audio_processor.process(args.audio1)
                inputs1 = wavlm_extractor(waveform1.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
                inputs1 = {k: v.to(device) for k, v in inputs1.items()}
                embedding1 = model(**inputs1).embeddings.squeeze().cpu()
                
                # Audio 2
                waveform2 = audio_processor.process(args.audio2)
                inputs2 = wavlm_extractor(waveform2.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
                inputs2 = {k: v.to(device) for k, v in inputs2.items()}
                embedding2 = model(**inputs2).embeddings.squeeze().cpu()
            else:
                # Audio 1
                waveform1 = audio_processor.process(args.audio1).unsqueeze(0).to(device)
                embedding1 = model.encode_batch(waveform1).squeeze().cpu()
                
                # Audio 2
                waveform2 = audio_processor.process(args.audio2).unsqueeze(0).to(device)
                embedding2 = model.encode_batch(waveform2).squeeze().cpu()
        else:
            # Audio 1
            waveform1 = audio_processor.process(args.audio1)
            features1 = feature_extractor.extract(waveform1).unsqueeze(0).to(device)
            embedding1 = model(features1, return_embedding=True).squeeze().cpu()
            
            # Audio 2
            waveform2 = audio_processor.process(args.audio2)
            features2 = feature_extractor.extract(waveform2).unsqueeze(0).to(device)
            embedding2 = model(features2, return_embedding=True).squeeze().cpu()
    
    # Compute similarity
    score = verifier.compute_score(embedding1, embedding2)
    accepted = score.item() >= threshold
    
    # Display results
    print(f"\n{'='*50}")
    print(f"Speaker Verification Result")
    print(f"{'='*50}")
    print(f"Audio 1: {args.audio1}")
    print(f"Audio 2: {args.audio2}")
    print(f"{'='*50}")
    print(f"Similarity Score: {score.item():.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Decision: {'[MATCH] SAME SPEAKER' if accepted else '[NO MATCH] DIFFERENT SPEAKERS'}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
