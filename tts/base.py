from abc import ABC, abstractmethod
from typing import Optional

class TTSModel(ABC):
    """Abstract base class for all TTS backends."""

    @abstractmethod
    def load(self, model_path: str, vocoder_path: Optional[str] = None):
        """
        Load model weights / checkpoint.
        
        Args:
            model_path: Path to TTS model checkpoint.
            vocoder_path: Optional path to vocoder checkpoint.
        """
        pass

    @abstractmethod
    def synthesize(self, text: str, speaker: Optional[str] = None, 
                   emotion: Optional[str] = None, use_vocoder: bool = True) -> bytes:
        """
        Convert text to raw audio bytes (e.g. WAV data).
        
        Args:
            text: Input text to speak.
            speaker: Optional speaker ID or name (if multi-speaker).
            emotion: Optional emotion label.
            use_vocoder: Whether to use the vocoder if loaded.
        
        Returns:
            bytes: Audio waveform (WAV or PCM data).
        """
        pass
