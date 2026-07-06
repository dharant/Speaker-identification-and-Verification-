# System Design Document

## 1. System Overview

The Speaker Identification System is a modular deep learning pipeline that processes raw audio signals to identify or verify speakers. The system is built with PyTorch and leverages the ECAPA-TDNN architecture for extracting discriminative speaker embeddings.

## 2. Architecture

### 2.1 High-Level Pipeline

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌────────────────┐
│  Raw Audio   │───▶│  Preprocessing   │───▶│  Feature Extract │───▶│   ECAPA-TDNN   │
│  (WAV/FLAC)  │    │  (16kHz, Norm)   │    │  (MFCC/Mel)     │    │  (Embedding)   │
└─────────────┘    └──────────────────┘    └──────────────────┘    └───────┬────────┘
                                                                           │
                                                              ┌────────────┴────────────┐
                                                              │                         │
                                                    ┌─────────▼─────────┐     ┌────────▼────────┐
                                                    │   Identification  │     │  Verification   │
                                                    │  (ArcFace Head)   │     │ (Cosine Score)  │
                                                    └─────────┬─────────┘     └────────┬────────┘
                                                              │                        │
                                                    ┌─────────▼─────────┐     ┌────────▼────────┐
                                                    │  Speaker Label    │     │ Accept/Reject   │
                                                    └───────────────────┘     └─────────────────┘
```

### 2.2 Module Descriptions

#### Audio Preprocessing (`src/preprocessing/audio_processor.py`)
- **Input**: Raw audio files (WAV, FLAC, MP3)
- **Operations**: 
  - Multi-format loading via torchaudio
  - Stereo-to-mono conversion
  - Resampling to 16kHz
  - Peak normalization
  - Energy-based silence trimming (30dB threshold)
  - Fixed-length output (3 seconds = 48,000 samples)
- **Output**: 1D tensor of shape `(48000,)`

#### Feature Extraction (`src/preprocessing/feature_extractor.py`)
- **Input**: Preprocessed waveform
- **MFCC Configuration**: 40 coefficients, 512-point FFT, 25ms window, 10ms hop
- **Mel Spectrogram**: 80 filterbank channels, same FFT/window/hop parameters
- **Output**: 2D tensor of shape `(40, 300)` for 3-second audio

#### Data Augmentation (`src/preprocessing/augmentation.py`)
Applied during training only:
- **Additive Noise**: White/pink noise at 5-20 dB SNR
- **Speed Perturbation**: 0.9x to 1.1x speed factors
- **Reverberation**: Synthetic impulse response convolution
- **SpecAugment**: Time and frequency masking on spectrograms

#### ECAPA-TDNN Model (`src/models/ecapa_tdnn.py`)
```
Input (40 × T) → Conv1D (512, k=5) → BN → ReLU
    → SE-Res2Net Block (512, k=3, d=2)  → out2
    → SE-Res2Net Block (512, k=3, d=3)  → out3
    → SE-Res2Net Block (512, k=3, d=4)  → out4
    → Cat[out2, out3, out4] (1536 × T)
    → Conv1D (1536, k=1) → BN → ReLU
    → Attentive Statistics Pooling → (3072)
    → BN → Linear (192) → BN
    → L2 Normalize → Speaker Embedding (192)
```

**Key innovations:**
- **Res2Net blocks**: Split channels into 8 sub-groups for multi-scale processing
- **SE (Squeeze-and-Excitation)**: Channel attention mechanism
- **Multi-layer Feature Aggregation**: Concatenate intermediate layer outputs
- **Attentive Statistics Pooling**: Attention-weighted mean + std pooling

#### Classification Heads (`src/models/classifier.py`)
- **ArcFace**: `cos(θ + m)` margin on target class, scale factor s=30
- **Softmax**: Standard linear classifier (for inference)

#### Verification Module (`src/models/verification.py`)
- Enrollment: Average multiple embeddings per speaker
- Scoring: Cosine similarity between test embedding and enrolled profile
- Decision: Compare score against threshold

## 3. Training Pipeline

### 3.1 Data Flow
```
Dataset → Balanced Sampler → Augmentation → Feature Extraction
    → ECAPA-TDNN → ArcFace Classifier → Cross-Entropy Loss → Backprop
```

### 3.2 Optimization
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler**: Cosine annealing with 5-epoch warmup
- **Mixed Precision**: FP16 with gradient scaling (GPU only)
- **Gradient Clipping**: Max norm 5.0

### 3.3 Checkpointing
- Regular checkpoints every 5 epochs
- Best model saved based on validation EER
- Full state preservation (model, optimizer, scheduler, history)

## 4. Evaluation Framework

### 4.1 Identification Evaluation
- Top-1 classification accuracy on test set
- Confusion matrix analysis for error patterns

### 4.2 Verification Evaluation
- Generate balanced positive/negative trial pairs
- Compute EER, minDCF, ROC AUC
- Score distribution analysis

### 4.3 Latency Measurement
- Warm-up runs to stabilize GPU state
- Statistics: mean, std, min, max, P95

## 5. Design Decisions

| Decision | Rationale |
|----------|-----------|
| ECAPA-TDNN over ResNet | Better performance on speaker verification benchmarks |
| ArcFace over Triplet Loss | More stable training, better convergence |
| 192-dim embeddings | Good balance of discriminability and efficiency |
| Cosine similarity scoring | Simple, effective, no additional training needed |
| Mixed precision training | 2x speedup on GPU with minimal accuracy loss |
| Mini dataset generator | Enables immediate testing without large dataset downloads |

## 6. Scalability Considerations

- **Multi-GPU**: DataParallel/DistributedDataParallel compatible
- **Batch Processing**: Variable-length collation with padding
- **Real-time Inference**: ~10-50ms per utterance on GPU
- **Memory Efficient**: Mixed precision reduces GPU memory by ~40%
