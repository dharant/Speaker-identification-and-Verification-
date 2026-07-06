"""
Speaker Verification Module
============================
Implements speaker verification using cosine similarity scoring
with enrollment and verification capabilities.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class SpeakerVerifier:
    """
    Speaker verification system using cosine similarity.
    
    Supports enrollment of speaker profiles and verification of
    test utterances against enrolled speakers.
    
    Args:
        threshold (float): Verification decision threshold. Pairs with
                          similarity above this threshold are accepted.
        scoring_method (str): Scoring method ("cosine" or "euclidean").
    """
    
    def __init__(self, threshold: float = 0.5, scoring_method: str = "cosine"):
        self.threshold = threshold
        self.scoring_method = scoring_method
        self.enrolled_speakers: Dict[str, torch.Tensor] = {}
    
    def enroll(
        self, speaker_id: str, embeddings: torch.Tensor, aggregate: bool = True
    ) -> None:
        """
        Enroll a speaker with one or more embedding vectors.
        
        Args:
            speaker_id (str): Unique speaker identifier.
            embeddings (torch.Tensor): Speaker embeddings of shape
                                       (n_utterances, embedding_dim) or (embedding_dim,).
            aggregate (bool): If True, average multiple embeddings into one profile.
        """
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        
        if aggregate:
            # Average embeddings and normalize
            profile = embeddings.mean(dim=0)
            profile = F.normalize(profile, p=2, dim=0)
        else:
            profile = F.normalize(embeddings, p=2, dim=1)
        
        self.enrolled_speakers[speaker_id] = profile
    
    def verify(
        self, embedding: torch.Tensor, speaker_id: str
    ) -> Tuple[bool, float]:
        """
        Verify if an utterance belongs to a claimed speaker.
        
        Args:
            embedding (torch.Tensor): Test utterance embedding of shape (embedding_dim,).
            speaker_id (str): Claimed speaker identity.
            
        Returns:
            Tuple[bool, float]: (accepted, similarity_score).
                                accepted is True if score >= threshold.
            
        Raises:
            KeyError: If speaker_id is not enrolled.
        """
        if speaker_id not in self.enrolled_speakers:
            raise KeyError(f"Speaker '{speaker_id}' is not enrolled.")
        
        profile = self.enrolled_speakers[speaker_id]
        embedding = F.normalize(embedding, p=2, dim=0)
        
        score = self.compute_score(embedding, profile)
        accepted = score >= self.threshold
        
        return accepted, score.item()
    
    def identify(
        self, embedding: torch.Tensor, top_k: int = 1
    ) -> list:
        """
        Identify a speaker from enrolled speakers.
        
        Args:
            embedding (torch.Tensor): Test utterance embedding.
            top_k (int): Number of top candidates to return.
            
        Returns:
            list: List of (speaker_id, score) tuples, sorted by score descending.
        """
        if not self.enrolled_speakers:
            return []
        
        embedding = F.normalize(embedding, p=2, dim=0)
        
        scores = []
        for speaker_id, profile in self.enrolled_speakers.items():
            score = self.compute_score(embedding, profile)
            scores.append((speaker_id, score.item()))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    def compute_score(
        self, embedding1: torch.Tensor, embedding2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity score between two embeddings.
        
        Args:
            embedding1 (torch.Tensor): First embedding.
            embedding2 (torch.Tensor): Second embedding.
            
        Returns:
            torch.Tensor: Similarity score (scalar).
        """
        if self.scoring_method == "cosine":
            if embedding1.dim() == 1 and embedding2.dim() == 1:
                return F.cosine_similarity(
                    embedding1.unsqueeze(0), embedding2.unsqueeze(0)
                ).squeeze()
            return F.cosine_similarity(embedding1, embedding2, dim=-1).mean()
        elif self.scoring_method == "euclidean":
            distance = torch.dist(embedding1, embedding2, p=2)
            # Convert distance to similarity (0 distance = 1 similarity)
            return 1.0 / (1.0 + distance)
        else:
            raise ValueError(f"Unknown scoring method: {self.scoring_method}")
    
    def compute_scores_batch(
        self, embeddings1: torch.Tensor, embeddings2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise similarity scores for batches of embeddings.
        
        Args:
            embeddings1 (torch.Tensor): First batch (n, embedding_dim).
            embeddings2 (torch.Tensor): Second batch (n, embedding_dim).
            
        Returns:
            torch.Tensor: Similarity scores of shape (n,).
        """
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        
        if self.scoring_method == "cosine":
            return F.cosine_similarity(embeddings1, embeddings2, dim=1)
        elif self.scoring_method == "euclidean":
            distances = torch.norm(embeddings1 - embeddings2, p=2, dim=1)
            return 1.0 / (1.0 + distances)
        else:
            raise ValueError(f"Unknown scoring method: {self.scoring_method}")
    
    def set_threshold(self, threshold: float) -> None:
        """Update the verification decision threshold."""
        self.threshold = threshold
    
    def get_enrolled_speakers(self) -> list:
        """Return a list of enrolled speaker IDs."""
        return list(self.enrolled_speakers.keys())
    
    def remove_speaker(self, speaker_id: str) -> None:
        """Remove an enrolled speaker."""
        if speaker_id in self.enrolled_speakers:
            del self.enrolled_speakers[speaker_id]
    
    def clear_enrollment(self) -> None:
        """Remove all enrolled speakers."""
        self.enrolled_speakers.clear()
