from piper import PiperVoice
from .base import TTSModel
from ..utils import ARPAconverter
import os
import time
import torch
class Piper(TTSModel):
    def __init__(self):
        self.tmw = None #Piper Model
    
    def load(self, model_path: str):
        self.tmw = PiperVoice.load(model_path, use_cuda=torch.cuda.is_available())
    
    def synthesize(self, text, speaker_id, torchmoji_text=None, superress=4, arpaconv=True, skip_sr=False) -> bytes:
        audio_pth = os.path.join("generated_audio", f"{int(time.time() * 1000)}_tacotron_gen.wav")
        if arpaconv:
            text = ARPAconverter(text)
        
        import wave
        with wave.open(audio_pth, "wb") as wf:
            self.tmw.synthesize_wav(text=text, wav_file=wf)
        
        return audio_pth