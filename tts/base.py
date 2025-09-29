
from abc import ABC, abstractmethod
class TTSModel(ABC):
    """Abstract base class for all TTS backends."""
    @abstractmethod
    def load(self, model_path: str):
        """Load model weights / checkpoint."""
        pass
    @abstractmethod
    def synthesize(self, text: str, speaker: str = None, emotion: str = None) -> bytes:
        """
        Convert text to raw audio bytes (e.g. WAV data).
        
        Args:
            text: Input text to speak.
            speaker: Optional speaker ID or name (if multi-speaker).
            emotion: Optional emotion label.
        
        Returns:
            bytes: Audio waveform (WAV or PCM data).
        """
        pass

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_CPU = DEVICE == "cpu"