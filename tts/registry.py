import json
from .tacotron import Tacotron2HIFI
from .piper import Piper
from .rvc import RVC
import os

MODEL_CLASSES = {
    "Tacotron2": Tacotron2HIFI,
    "VITS": Piper,
    "RVC": RVC,
}

CONFIG_PATH = r"C:\Users\L\Desktop\41.ai-rewrite\tts\main_config.json"

with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG_PATH = json.load(f)["models"]

_loaded_models = {}

def get_model(model_name: str, extra_settings: dict):
    if extra_settings is None:
        extra_settings = {}
    if model_name not in CONFIG_PATH:
        raise ValueError(f"Unknown model: {model_name}")
    
    if model_name not in _loaded_models:
        cfg = CONFIG_PATH[model_name]
        arch = cfg["Architecture"]
        cls = MODEL_CLASSES[arch]
        model = cls()
        if arch == "VITS" or arch=="RVC":
            model.load(cfg["Checkpoint"], **extra_settings) 
        else:
            model.load(cfg["Checkpoint"], cfg["Vocoder"], **extra_settings) 
        _loaded_models[model_name] = model
    
    return _loaded_models[model_name]


#########For Front End#################
def list_models():
    """Return dict of all available models from config.json"""
    return CONFIG_PATH

def list_speakers(model_name: str):
    """Return list of speaker IDs for a specific model."""
    models = list_models()
    model = models.get(model_name)
    if not model:
        return []

    spk_path = model.get("SpeakerIDS")
    if not spk_path or not os.path.exists(spk_path):
        return []

    with open(spk_path, "r", encoding="utf-8") as f:
        speakers = [line.strip() for line in f if line.strip()]

    return speakers
