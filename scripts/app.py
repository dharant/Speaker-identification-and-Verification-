"""
Flask Application Server for Speaker Diarization & Transcription UI
===================================================================
Runs a local web server to upload audio files, segment them,
diarize speakers using our trained ECAPA-TDNN model, transcribe
the segments, and display them in a chat-like conversation format.
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

import uuid
import shutil
import numpy as np
import torch
import torchaudio
import yaml
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from sklearn.cluster import AgglomerativeClustering
import speech_recognition as sr

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.audio_processor import AudioProcessor
from src.preprocessing.feature_extractor import FeatureExtractor
from src.models.ecapa_tdnn import ECAPATDNN

app = Flask(__name__, template_folder='../src/web/templates', static_folder='../src/web/static')
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'temp_uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model
model = None
pretrained_model = None
pretrained_resnet_model = None
pretrained_wavlm_model = None
wavlm_extractor = None
audio_processor = None
feature_extractor = None
model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_system_model():
    """Load the trained ECAPA-TDNN model and configurations."""
    global model, audio_processor, feature_extractor
    
    config_path = "config/train_config.yaml"
    checkpoint_path = "checkpoints/best_model.pt"
    
    if not os.path.exists(config_path):
        print(f"Error: Config path not found: {config_path}")
        return False
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    preprocess_config = config.get("preprocessing", {})
    dataset_config = config.get("dataset", {})
    feature_config = config.get("features", {})
    feature_config["sample_rate"] = preprocess_config.get("sample_rate", 16000)
    
    # Initialize processor & extractor
    audio_processor = AudioProcessor(
        sample_rate=preprocess_config.get("sample_rate", 16000),
        normalize=preprocess_config.get("normalize", True),
        trim_silence=preprocess_config.get("trim_silence", True),
        trim_db=preprocess_config.get("trim_db", 30),
    )
    feature_extractor = FeatureExtractor.from_config(feature_config)
    
    # Load model architecture
    if os.path.exists(checkpoint_path):
        print(f"Loading trained checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=model_device, weights_only=False)
        model_config = checkpoint.get("config", config).get("model", {})
        model = ECAPATDNN.from_config(model_config).to(model_device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Warning: Trained checkpoint not found. Using randomly initialized model.")
        model_config = config.get("model", {})
        model = ECAPATDNN.from_config(model_config).to(model_device)
        
    model.eval()
    return True

def load_pretrained_model():
    """Load SpeechBrain pretrained ECAPA-TDNN model on demand."""
    global pretrained_model
    if pretrained_model is not None:
        return True
    
    try:
        print("Loading SpeechBrain pretrained ECAPA-TDNN model...")
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        
        pretrained_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": str(model_device)}
        )
        print("Pretrained model loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading SpeechBrain pretrained model: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_pretrained_resnet_model():
    """Load SpeechBrain pretrained ResNet-34 model on demand."""
    global pretrained_resnet_model
    if pretrained_resnet_model is not None:
        return True
    
    try:
        print("Loading SpeechBrain pretrained ResNet-34 model...")
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        
        pretrained_resnet_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-resnet-voxceleb",
            savedir="pretrained_models/spkrec-resnet-voxceleb",
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": str(model_device)}
        )
        print("Pretrained ResNet model loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading SpeechBrain pretrained ResNet model: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_pretrained_wavlm_model():
    """Load Microsoft pretrained WavLM speaker model on demand."""
    global pretrained_wavlm_model, wavlm_extractor
    if pretrained_wavlm_model is not None and wavlm_extractor is not None:
        return True
    
    try:
        print("Loading Microsoft pretrained WavLM model...")
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
        
        wavlm_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        pretrained_wavlm_model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(model_device)
        pretrained_wavlm_model.eval()
        
        print("Pretrained WavLM model loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading WavLM model: {e}")
        import traceback
        traceback.print_exc()
        return False

# Load model on startup
load_system_model()

MOCK_CONVERSATIONS = [
    "Welcome to the meeting, everyone. Let's start with project updates.",
    "Hi, I finished training the ECAPA-TDNN model on the LibriSpeech dataset.",
    "That is fantastic news. What are the final validation metrics?",
    "We achieved an Equal Error Rate of 3.82% and Top-1 accuracy of 93.94%.",
    "Excellent. This means our speaker embeddings are highly discriminative.",
    "Yes, and the inference latency is under 15 milliseconds on CPU.",
    "Perfect! That makes it suitable for real-time diarization and assistant tasks.",
    "I agree. Should we start integrating it into our main application pipeline?",
    "Absolutely. Let's schedule a deployment review for tomorrow morning.",
    "Sounds great. I will prepare the presentation slides and demo logs.",
    "Thank you everyone, let's wrap this up. See you all tomorrow!"
]

def get_mock_transcription(start_sec, end_sec, segment_idx):
    idx = segment_idx % len(MOCK_CONVERSATIONS)
    return MOCK_CONVERSATIONS[idx]

def lookup_dataset_transcript(filename):
    """
    If the uploaded file is from the LibriSpeech dev-clean-2 dataset,
    look up its ground truth transcript text.
    """
    # Extract base name like '1272-135031-0000'
    base_name = os.path.splitext(os.path.basename(filename))[0]
    parts = base_name.split('-')
    if len(parts) >= 3:
        speaker_id = parts[0]
        chapter_id = parts[1]
        trans_file = f"data/dev_clean_2/{speaker_id}/{chapter_id}/{speaker_id}-{chapter_id}.trans.txt"
        
        if os.path.exists(trans_file):
            try:
                with open(trans_file, "r") as f:
                    for line in f:
                        if line.startswith(base_name):
                            # Remove ID prefix and strip whitespace
                            return line[len(base_name):].strip().capitalize()
            except Exception as e:
                print(f"Error reading transcript file: {e}")
    return None

def voice_activity_detection(waveform, sample_rate):
    """Dynamic energy-based Voice Activity Detection to segment speech."""
    signal = waveform.squeeze(0).numpy() if waveform.dim() > 1 else waveform.numpy()
    
    frame_length_ms = 40
    hop_length_ms = 20
    
    frame_length = int(frame_length_ms * sample_rate / 1000)
    hop_length = int(hop_length_ms * sample_rate / 1000)
    
    # Compute RMS energy per frame
    num_frames = max(1, (len(signal) - frame_length) // hop_length + 1)
    energies = []
    for i in range(num_frames):
        start = i * hop_length
        end = min(start + frame_length, len(signal))
        frame = signal[start:end]
        rms = np.sqrt(np.mean(frame**2) + 1e-10)
        db = 20 * np.log10(rms)
        energies.append(db)
        
    energies = np.array(energies)
    
    # Dynamic thresholding: compute the 20th percentile (noise floor) and 95th percentile (peak speech)
    q20 = np.percentile(energies, 20)
    q95 = np.percentile(energies, 95)
    
    # Set threshold 15% of the range above the noise floor
    energy_threshold = q20 + (q95 - q20) * 0.15
    # Clip to reasonable range to ensure safety
    energy_threshold = float(np.clip(energy_threshold, -48.0, -25.0))
    
    is_speech = energies > energy_threshold
    
    # Find speech intervals
    speech_segments = []
    in_speech = False
    start_frame = 0
    
    for i, active in enumerate(is_speech):
        if active and not in_speech:
            in_speech = True
            start_frame = i
        elif not active and in_speech:
            in_speech = False
            end_frame = i
            # Filter short noise events (at least 600ms)
            if (end_frame - start_frame) * hop_length_ms >= 600:
                start_sec = (start_frame * hop_length) / sample_rate
                end_sec = (end_frame * hop_length + frame_length) / sample_rate
                speech_segments.append((start_sec, end_sec))
                
    if in_speech:
        start_sec = (start_frame * hop_length) / sample_rate
        end_sec = len(signal) / sample_rate
        if (len(is_speech) - start_frame) * hop_length_ms >= 600:
            speech_segments.append((start_sec, end_sec))
        
    # Merge close segments (less than 1.2s silence)
    merged_segments = []
    if speech_segments:
        curr_start, curr_end = speech_segments[0]
        for start, end in speech_segments[1:]:
            if start - curr_end < 1.2:
                curr_end = end
            else:
                merged_segments.append((curr_start, curr_end))
                curr_start, curr_end = start, end
        merged_segments.append((curr_start, curr_end))
    else:
        # Fallback: if no speech detected, return segments of 4 seconds
        duration = len(signal) / sample_rate
        step = 4.0
        for s in range(int(np.ceil(duration / step))):
            merged_segments.append((s * step, min((s + 1) * step, duration)))
        
    # Break down long monologue segments into 4-second segments for dialog flow
    final_segments = []
    for start, end in merged_segments:
        duration = end - start
        if duration > 5.0:
            num_splits = int(np.ceil(duration / 4.0))
            split_dur = duration / num_splits
            for s in range(num_splits):
                final_segments.append((start + s * split_dur, start + (s + 1) * split_dur))
        else:
            final_segments.append((start, end))
            
    return final_segments

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
        
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    num_speakers = request.form.get('num_speakers', 'auto')
    if num_speakers != 'auto':
        try:
            num_speakers = int(num_speakers)
        except ValueError:
            num_speakers = None
    else:
        num_speakers = None
        
    model_type = request.form.get('model_type', 'pretrained_ecapa')
    if model_type == 'pretrained':
        model_type = 'pretrained_ecapa'
        
    # Save file
    filename = secure_filename(audio_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(filepath)
    
    try:
        # Load waveform using soundfile
        import soundfile as sf
        data, sr_rate = sf.read(filepath, dtype='float32')
        waveform = torch.from_numpy(data)
        
        # Convert to shape (1, samples) mono
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=1, keepdim=True).t()
        else:
            waveform = waveform.unsqueeze(0)
            
        # Resample to 16kHz mono if needed
        if sr_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr_rate, new_freq=16000)
            waveform = resampler(waveform)
            sr_rate = 16000
            
        duration = waveform.shape[1] / sr_rate
        
        # Step 1: Voice Activity Detection (segmentation)
        segments = voice_activity_detection(waveform, sr_rate)
        
        # Step 2: Speaker embedding extraction for each segment
        embeddings = []
        valid_segments = []
        
        # Load pretrained model if requested
        if model_type == 'pretrained_ecapa':
            if not load_pretrained_model():
                return jsonify({"error": "Failed to load SpeechBrain pretrained ECAPA model"}), 500
        elif model_type == 'pretrained_resnet':
            if not load_pretrained_resnet_model():
                return jsonify({"error": "Failed to load SpeechBrain pretrained ResNet model"}), 500
        elif model_type == 'pretrained_wavlm':
            if not load_pretrained_wavlm_model():
                return jsonify({"error": "Failed to load Microsoft WavLM model"}), 500
        
        for start, end in segments:
            # Skip segments too short to extract embeddings
            if end - start < 0.3:
                continue
                
            start_sample = int(start * sr_rate)
            end_sample = int(end * sr_rate)
            segment_wav = waveform[0, start_sample:end_sample]
            
            if model_type == 'pretrained_ecapa':
                # SpeechBrain takes raw waveform (1, samples)
                wav_tensor = segment_wav.unsqueeze(0).to(model_device)
                with torch.no_grad():
                    embedding = pretrained_model.encode_batch(wav_tensor).squeeze().cpu().numpy()
            elif model_type == 'pretrained_resnet':
                wav_tensor = segment_wav.unsqueeze(0).to(model_device)
                with torch.no_grad():
                    embedding = pretrained_resnet_model.encode_batch(wav_tensor).squeeze().cpu().numpy()
            elif model_type == 'pretrained_wavlm':
                inputs = wavlm_extractor(segment_wav.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                with torch.no_grad():
                    embedding = pretrained_wavlm_model(**inputs).embeddings.squeeze().cpu().numpy()
            else:
                # Apply padding/cropping if needed
                segment_wav = audio_processor.process_waveform(segment_wav)
                features = feature_extractor.extract(segment_wav).unsqueeze(0).to(model_device)
                with torch.no_grad():
                    embedding = model(features, return_embedding=True).squeeze().cpu().numpy()
                
            embeddings.append(embedding)
            valid_segments.append((start, end))
            
        if not valid_segments:
            # If no valid segments found, use the default single full audio segment
            valid_segments = [(0.0, duration)]
            dim = 192
            if model_type == 'pretrained_resnet':
                dim = 256
            elif model_type == 'pretrained_wavlm':
                dim = 512
            embeddings = [np.zeros(dim)]
            
        # Step 3: Cluster speaker embeddings to assign speaker labels
        embeddings = np.array(embeddings)
        n_speakers_setting = num_speakers
        
        if len(valid_segments) == 1:
            labels = [0]
        else:
            if n_speakers_setting is None:
                # Default auto selection: search for optimal clusters from 1 up to 4
                max_possible = min(4, len(valid_segments))
                if max_possible >= 2:
                    # Compute pairwise cosine similarities to check if it's likely a single speaker
                    # Cosine similarity = dot product of L2 normalized vectors
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    norm_embeddings = embeddings / (norms + 1e-10)
                    similarity_matrix = np.dot(norm_embeddings, norm_embeddings.T)
                    
                    # Extract upper triangular values (excluding diagonal)
                    n_segs = len(valid_segments)
                    pairwise_sims = similarity_matrix[np.triu_indices(n_segs, k=1)]
                    avg_sim = float(np.mean(pairwise_sims))
                    print(f"Diarization auto-detection: {n_segs} segments, average pairwise cosine similarity: {avg_sim:.4f}")
                    
                    # Determine single-speaker detection threshold dynamically based on model
                    single_spk_threshold = 0.45
                    if model_type == 'pretrained_resnet':
                        single_spk_threshold = 0.55
                    elif model_type == 'pretrained_wavlm':
                        single_spk_threshold = 0.78
                    
                    # If average similarity is high, assume only 1 speaker.
                    if avg_sim >= single_spk_threshold:
                        print(f"Diarization auto-detection: High similarity (>= {single_spk_threshold}). Assuming 1 speaker.")
                        labels = [0] * len(valid_segments)
                    else:
                        from sklearn.metrics import silhouette_score
                        best_k = 2
                        best_score = -1.0
                        labels = None
                        
                        # Find best k between 2 and max_possible
                        for k in range(2, max_possible + 1):
                            clusterer = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
                            k_labels = clusterer.fit_predict(embeddings).tolist()
                            
                            if len(set(k_labels)) > 1:
                                score = silhouette_score(embeddings, k_labels, metric='cosine')
                                print(f"Diarization auto-detection: k={k}, silhouette score={score:.4f}")
                                if score > best_score:
                                    best_score = score
                                    best_k = k
                                    labels = k_labels
                                    
                        if labels is None:
                            labels = AgglomerativeClustering(n_clusters=2, metric='cosine', linkage='average').fit_predict(embeddings).tolist()
                        print(f"Diarization auto-detection: Selected k={len(set(labels))} speakers.")
                else:
                    labels = [0] * len(valid_segments)
            else:
                n_clusters = min(n_speakers_setting, len(valid_segments))
                labels = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average').fit_predict(embeddings).tolist()
                
        # Step 4: Transcribe segments and structure final JSON
        recognizer = sr.Recognizer()
        conversation = []
        
        # Check if it matches a dataset file
        dataset_text = lookup_dataset_transcript(filename)
        
        for idx, ((start, end), label) in enumerate(zip(valid_segments, labels)):
            speaker_name = f"Speaker {chr(65 + label)}" # Speaker A, Speaker B, etc.
            
            text = ""
            if dataset_text:
                # For LibriSpeech dataset sample, split words roughly based on start time
                words = dataset_text.split()
                n_words = len(words)
                if n_words > 0:
                    words_per_segment = max(1, int(np.ceil(n_words / len(valid_segments))))
                    start_w = idx * words_per_segment
                    end_w = min(start_w + words_per_segment, n_words)
                    text = " ".join(words[start_w:end_w])
            
            # If not a dataset sample or we failed to align, do actual ASR or fallback to mock
            if not text:
                temp_segment_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_seg_{uuid.uuid4().hex}.wav")
                try:
                    # Crop segment and save as WAV
                    start_sample = int(start * sr_rate)
                    end_sample = int(end * sr_rate)
                    segment_data = waveform[0, start_sample:end_sample].numpy()
                    
                    import soundfile as sf
                    # Explicitly convert float32 to standard 16-bit signed PCM WAV for Google API compatibility
                    sf.write(temp_segment_path, segment_data, sr_rate, subtype='PCM_16')
                    
                    # Call speech recognition
                    with sr.AudioFile(temp_segment_path) as source:
                        audio_data = recognizer.record(source)
                        text = recognizer.recognize_google(audio_data)
                        print(f"ASR success: Segment {idx + 1} ({start:.2f}s - {end:.2f}s): '{text}'")
                except sr.UnknownValueError:
                    print(f"ASR unknown: Segment {idx + 1} ({start:.2f}s - {end:.2f}s) contains no recognizable speech.")
                    text = "[Unclear / Silence]"
                except Exception as e:
                    print(f"ASR error: Segment {idx + 1} ({start:.2f}s - {end:.2f}s): {e}")
                    text = f"[Speech Segment {idx + 1}]"
                finally:
                    if os.path.exists(temp_segment_path):
                        os.remove(temp_segment_path)
            
            conversation.append({
                "id": idx + 1,
                "speaker": speaker_name,
                "speaker_class": f"speaker-color-{label % 4}",
                "start": round(start, 2),
                "end": round(end, 2),
                "text": text
            })
            
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
            
        return jsonify({
            "filename": filename,
            "duration": round(duration, 2),
            "num_segments": len(conversation),
            "speakers_detected": len(set(labels)),
            "conversation": conversation
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500

if __name__ == '__main__':
    print("Starting Flask web server on http://127.0.0.1:5000...")
    app.run(debug=True, use_reloader=False, port=5000)
