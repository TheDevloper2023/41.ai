import json
from .tacotron import Tacotron2
from .piper import Piper

MODEL_CLASSES = {
    "Tacotron2": Tacotron2,
    "VITS": Piper,
}

with open("C:/Users/L/Desktop/41.ai rewrite/main_config.json") as f:
    MODEL_CONFIG = json.load(f)["models"]

_loaded_models = {}

def get_model(model_name: str):
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Unknown model: {model_name}")
    
    if model_name not in _loaded_models:
        cfg = MODEL_CONFIG[model_name]
        arch = cfg["Architecture"]
        cls = MODEL_CLASSES[arch]
        model = cls()
        model.load(cfg["Checkpoint"])  # pass in checkpoint
        _loaded_models[model_name] = model
    
    return _loaded_models[model_name]