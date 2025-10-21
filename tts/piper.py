from piper import PiperVoice, SynthesisConfig
from .base import TTSModel
import os
import time
import torch
class Piper(TTSModel):
    def __init__(self):
        self.tmw = None #Piper Model
    
    def load(self, model_path: str, **extra_settings):
        self.tmw = PiperVoice.load(model_path, use_cuda=torch.cuda.is_available())
    
    def synthesize(self, text, speaker_id, torchmoji_text=None, **extra_settings) -> bytes:
        audio_pth = os.path.join("generated_audio", f"{int(time.time() * 1000)}_tacotron_gen.wav")
        

        conf = SynthesisConfig( #Adding speaker selection, More expressive shit.
        speaker_id=speaker_id,
        noise_scale=1.0,
        noise_w_scale=1.0
        
        )

        import wave
        with wave.open(audio_pth, "wb") as wf:
            self.tmw.synthesize_wav(text=text, wav_file=wf, syn_config=conf)
        
        return audio_pth