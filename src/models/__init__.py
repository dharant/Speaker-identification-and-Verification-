"""Model architectures for speaker identification."""

from .ecapa_tdnn import ECAPATDNN
from .classifier import ArcFaceClassifier, SoftmaxClassifier
from .verification import SpeakerVerifier

__all__ = ["ECAPATDNN", "ArcFaceClassifier", "SoftmaxClassifier", "SpeakerVerifier"]
