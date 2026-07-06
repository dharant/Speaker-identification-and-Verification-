# Speaker Identification System

A deep learning-based speaker identification and verification system using **PyTorch** and **SpeechBrain**, implementing the ECAPA-TDNN architecture for robust speaker embedding extraction.

## 🎯 Project Overview

This system leverages state-of-the-art deep learning techniques to:
- **Identify speakers** (closed-set multi-class classification) from audio recordings
- **Verify speakers** (open-set binary classification) by comparing voice embeddings
- Extract discriminative **speaker embeddings** suitable for downstream applications

### Applications
- Voice biometric authentication
- Call center speaker analytics
- Personalized user experiences
- Access control systems
- Speaker diarization

## 🏗️ Architecture

```
Raw Audio → Preprocessing → Feature Extraction → ECAPA-TDNN → Speaker Embedding
                                                                    ↓
                                                    ┌───────────────┼───────────────┐
                                                    ↓                               ↓
                                            Identification                   Verification
                                         (ArcFace Softmax)            (Cosine Similarity)
                                                    ↓                               ↓
                                            Speaker ID                    Accept/Reject
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Audio Preprocessing** | Loading, resampling (16kHz), normalization, silence trimming |
| **Feature Extraction** | MFCC (40 coefficients) or Mel Spectrogram (80 bins) |
| **Data Augmentation** | Noise addition, speed perturbation, reverberation, SpecAugment |
| **ECAPA-TDNN** | SE-Res2Net blocks + attentive statistics pooling → 192-dim embeddings |
| **ArcFace Classifier** | Additive Angular Margin Softmax for discriminative training |
| **Speaker Verifier** | Cosine similarity scoring with configurable threshold |

## 📋 Prerequisites

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for full training; CPU works for demo)

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd speaker-identification-system
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 📊 Dataset

### Option 1: Mini Dataset (Quick Demo)

Generate a synthetic mini dataset for testing:

```bash
python scripts/generate_mini_dataset.py --speakers 10 --utterances 50
```

This creates a small dataset with 10 speakers × 50 utterances each.

### Option 2: LibriSpeech Dataset (Recommended for Standard/Full Training)

You can download either a small subset or the full validation set of LibriSpeech for training:

**To download the mini subset (~15MB, 26 speakers, quick start):**
```bash
python scripts/download_librispeech.py --dataset dev-clean-2
```

**To download the standard/full subset (~337MB, 40 speakers, recommended for higher accuracy):**
```bash
python scripts/download_librispeech.py --dataset dev-clean
```

After downloading, update `config/train_config.yaml` to point to the desired dataset path:
```yaml
dataset:
  name: "dev_clean"                      # "dev_clean" or "dev_clean_2"
  root_dir: "data/dev_clean"             # "data/dev_clean" or "data/dev_clean_2"
```

### Option 3: VoxCeleb Dataset (Advanced Full Training)

1. Register at [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
2. Download VoxCeleb1 or VoxCeleb2
3. Extract to `data/voxceleb1/` or `data/voxceleb2/`
4. Update `config/train_config.yaml`:

```yaml
dataset:
  name: "voxceleb1"
  root_dir: "data/voxceleb1/wav"
```

## 🏋️ Training

### Quick Start (Mini Dataset)

```bash
# Generate dataset and train
python scripts/train.py --config config/train_config.yaml --generate-dataset --epochs 20
```

### Full Training

```bash
# Standard training
python scripts/train.py --config config/train_config.yaml

# Resume from checkpoint
python scripts/train.py --config config/train_config.yaml --resume checkpoints/checkpoint_epoch_10.pt

# Custom settings
python scripts/train.py --config config/train_config.yaml --epochs 100 --batch-size 64
```

### Training Configuration

All hyperparameters are in `config/train_config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Learning Rate | 0.001 | Initial learning rate |
| Batch Size | 32 | Training batch size |
| Epochs | 50 | Number of training epochs |
| Optimizer | Adam | Optimizer type |
| LR Scheduler | Cosine | Learning rate schedule |
| ArcFace Margin | 0.2 | Angular margin for ArcFace |
| Embedding Dim | 192 | Speaker embedding dimension |

## 📈 Evaluation

```bash
# Run full evaluation
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt

# Custom evaluation
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --num-trials 10000 --output-dir results
```

### Metrics Computed

| Metric | Description |
|--------|-------------|
| **Identification Accuracy** | Top-1 accuracy for closed-set identification |
| **EER** | Equal Error Rate for verification |
| **minDCF** | Minimum Detection Cost Function |
| **ROC AUC** | Area Under the ROC Curve |
| **Inference Latency** | Processing time per utterance |
| **Confusion Matrix** | Speaker-wise classification analysis |

## 🎤 Inference

### Speaker Identification

```bash
python scripts/identify.py --audio path/to/audio.wav --checkpoint checkpoints/best_model.pt --top-k 5
```

### Speaker Verification

```bash
python scripts/verify.py --audio1 path/to/file1.wav --audio2 path/to/file2.wav --checkpoint checkpoints/best_model.pt
```

### Web Application (Diarization & Transcription UI)

The system includes a premium web interface for visualizing speech segments, assigning speaker labels dynamically, and transcribing voice data in a conversation style.

To start the local web application server:

```bash
python scripts/app.py
```

Once running, navigate to the following URL in your web browser:
```
http://127.0.0.1:5000/
```

## 📁 Project Structure

```
├── config/
│   ├── train_config.yaml        # Training hyperparameters
│   └── inference_config.yaml    # Inference settings
├── src/
│   ├── preprocessing/
│   │   ├── audio_processor.py   # Audio loading & preprocessing
│   │   ├── feature_extractor.py # MFCC / Mel Spectrogram extraction
│   │   └── augmentation.py      # Data augmentation techniques
│   ├── models/
│   │   ├── ecapa_tdnn.py        # ECAPA-TDNN architecture
│   │   ├── classifier.py       # ArcFace / Softmax heads
│   │   └── verification.py     # Cosine similarity verification
│   ├── data/
│   │   ├── dataset.py           # PyTorch Dataset
│   │   ├── dataloader.py        # Custom DataLoader
│   │   └── mini_dataset.py      # Synthetic dataset generator
│   ├── training/
│   │   ├── losses.py            # ArcFace, Triplet, CE losses
│   │   └── trainer.py           # Training loop
│   ├── evaluation/
│   │   ├── metrics.py           # EER, DCF, Accuracy, ROC
│   │   └── evaluator.py         # Evaluation pipeline
│   └── inference/
│       └── pipeline.py          # End-to-end inference
├── scripts/
│   ├── train.py                 # Training entry point
│   ├── evaluate.py              # Evaluation entry point
│   ├── identify.py              # Speaker identification CLI
│   ├── verify.py                # Speaker verification CLI
│   └── generate_mini_dataset.py # Dataset generation
├── docs/
│   ├── system_design.md         # System architecture docs
│   └── evaluation_report.md     # Evaluation results
├── requirements.txt
├── .gitignore
└── README.md
```

## 🧪 Solution Approach

### Model: ECAPA-TDNN

The system uses **ECAPA-TDNN** (Emphasized Channel Attention, Propagation and Aggregation in TDNN), which achieves state-of-the-art results in speaker verification:

1. **SE-Res2Net Blocks**: Multi-scale feature extraction with channel attention
2. **Multi-layer Feature Aggregation**: Concatenates outputs from multiple layers
3. **Attentive Statistics Pooling**: Attention-weighted mean and standard deviation pooling
4. **192-dimensional Embeddings**: Compact speaker representations

### Training Strategy

- **ArcFace Loss**: Additive angular margin enforces inter-speaker separability
- **Cosine Annealing**: Learning rate schedule with warmup
- **Data Augmentation**: Noise, speed perturbation, reverberation, SpecAugment
- **Mixed Precision**: FP16 training for GPU efficiency

### Evaluation

- **Closed-set Identification**: Standard classification accuracy
- **Open-set Verification**: EER-based evaluation with cosine similarity scoring
- **Robustness Analysis**: Performance under varying noise and duration conditions

## ⚖️ Ethical Considerations

- This system processes voice biometric data, which is sensitive personal information
- Always obtain proper consent before recording and processing voice data
- Voice data should be stored securely with appropriate access controls
- The system should not be used for surveillance without proper legal authorization
- Bias evaluation across different demographics is recommended before deployment

## 📄 License

This project is for educational and research purposes. All dependencies are open-source.

## 📚 References

1. Desplanques, B., et al. (2020). "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification." *Proc. Interspeech*.
2. Deng, J., et al. (2019). "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." *CVPR*.
3. Nagrani, A., et al. (2017). "VoxCeleb: A Large-Scale Speaker Identification Dataset." *Proc. Interspeech*.
4. Ravanelli, M., et al. (2021). "SpeechBrain: A General-Purpose Speech Toolkit." *arXiv preprint*.
