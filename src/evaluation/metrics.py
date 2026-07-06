"""
Evaluation Metrics Module
=========================
Implements standard speaker recognition evaluation metrics:
- Equal Error Rate (EER)
- Detection Cost Function (DCF)
- Identification Accuracy (Top-1 and Top-K)
- ROC Curve and AUC
"""

import numpy as np
from typing import List, Tuple, Optional
from sklearn.metrics import roc_curve, auc, confusion_matrix


def compute_eer(
    scores: List[float], labels: List[int]
) -> float:
    """
    Compute the Equal Error Rate (EER).
    
    EER is the point where the False Acceptance Rate (FAR) equals
    the False Rejection Rate (FRR).
    
    Args:
        scores (list): Similarity scores for each trial.
        labels (list): Binary labels (1 = same speaker, 0 = different speaker).
        
    Returns:
        float: Equal Error Rate (0 to 1).
    """
    scores = np.array(scores)
    labels = np.array(labels)
    
    if len(np.unique(labels)) < 2:
        return 0.5  # Cannot compute EER without both classes
    
    # Compute FPR and TPR using sklearn
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    # Find the threshold where FPR ≈ FNR
    # EER is where the FPR and FNR curves intersect
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    
    return float(eer)


def compute_eer_with_threshold(
    scores: List[float], labels: List[int]
) -> Tuple[float, float]:
    """
    Compute EER and the corresponding threshold.
    
    Args:
        scores (list): Similarity scores.
        labels (list): Binary labels.
        
    Returns:
        Tuple[float, float]: (EER, threshold).
    """
    scores = np.array(scores)
    labels = np.array(labels)
    
    if len(np.unique(labels)) < 2:
        return 0.5, 0.5
    
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    eer_threshold = thresholds[eer_index] if eer_index < len(thresholds) else 0.5
    
    return float(eer), float(eer_threshold)


def compute_dcf(
    scores: List[float],
    labels: List[int],
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    """
    Compute the minimum Detection Cost Function (minDCF).
    
    DCF is a weighted combination of miss and false alarm rates,
    commonly used in NIST speaker recognition evaluations.
    
    Args:
        scores (list): Similarity scores.
        labels (list): Binary labels (1 = target, 0 = non-target).
        p_target (float): Prior probability of target speaker.
        c_miss (float): Cost of miss (false rejection).
        c_fa (float): Cost of false alarm (false acceptance).
        
    Returns:
        float: Minimum Detection Cost Function value.
    """
    scores = np.array(scores)
    labels = np.array(labels)
    
    if len(np.unique(labels)) < 2:
        return 1.0
    
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    # Compute DCF at each threshold
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    
    # Normalize by the best possible cost without the system
    default_dcf = min(c_miss * p_target, c_fa * (1 - p_target))
    
    if default_dcf > 0:
        min_dcf = np.min(dcf) / default_dcf
    else:
        min_dcf = np.min(dcf)
    
    return float(min_dcf)


def compute_accuracy(
    predictions: List[int],
    labels: List[int],
    top_k: int = 1,
) -> float:
    """
    Compute top-K identification accuracy.
    
    Args:
        predictions (list): Predicted labels or list of top-K predictions.
        labels (list): Ground truth labels.
        top_k (int): K value for top-K accuracy.
        
    Returns:
        float: Accuracy as a percentage (0-100).
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    if top_k == 1:
        correct = np.sum(predictions == labels)
    else:
        # predictions should be (n_samples, k) for top-k
        if predictions.ndim == 1:
            correct = np.sum(predictions == labels)
        else:
            correct = np.sum(
                np.any(predictions == labels.reshape(-1, 1), axis=1)
            )
    
    accuracy = 100.0 * correct / len(labels)
    return float(accuracy)


def compute_roc_auc(
    scores: List[float], labels: List[int]
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve and Area Under Curve (AUC).
    
    Args:
        scores (list): Similarity scores.
        labels (list): Binary labels.
        
    Returns:
        Tuple[ndarray, ndarray, float]: (FPR array, TPR array, AUC value).
    """
    scores = np.array(scores)
    labels = np.array(labels)
    
    if len(np.unique(labels)) < 2:
        return np.array([0, 1]), np.array([0, 1]), 0.5
    
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, float(roc_auc)


def compute_confusion_matrix(
    predictions: List[int], labels: List[int], num_classes: int = None
) -> np.ndarray:
    """
    Compute confusion matrix for speaker identification.
    
    Args:
        predictions (list): Predicted speaker labels.
        labels (list): Ground truth labels.
        num_classes (int): Number of classes (auto-detected if None).
        
    Returns:
        np.ndarray: Confusion matrix of shape (num_classes, num_classes).
    """
    if num_classes is not None:
        return confusion_matrix(
            labels, predictions, labels=list(range(num_classes))
        )
    return confusion_matrix(labels, predictions)
