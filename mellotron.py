import torch
from .base import TTSModel

class Mellotron(TTSModel):
    def __init__(self):
        self.ttm = None
        self.mtw = None
    

    def load(self, model_path: str, vocoder_path: str = None, **extra_settings):