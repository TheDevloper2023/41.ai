import json
from .tacotron import Tacotron2HIFI
from .piper import Piper
from .rvc import RVC
import os
from huggingface_hub import hf_hub_download
from pathlib import Path

MODEL_CLASSES = {
    "Tacotron2": Tacotron2HIFI,
    "VITS": Piper,
    "RVC": RVC,
}

CONFIG_PATH = "main_config_hf.json"

with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG_PATH = json.load(f)["models"]

_loaded_models = {}
last_used = {}
CACHE_DIR = ""
REPO_ID = "ViligerANON/41.ai_models" #It won't return shit for you as this is private
                                     # If you want sum public models, check https://huggingface.co/ViligerANON/Pipeline_Models/tree/main


def get_model(model_name: str, extra_settings: dict = None):
    extra_settings = extra_settings or {}
    
    # 1. Check if model is already in RAM
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    # 2. Get config for this specific voice
    cfg = CONFIG_PATH.get(model_name)
    if not cfg:
        raise ValueError(f"Model {model_name} not found in config.")

    print(f"--- Loading {model_name} from Hugging Face ---")

    # 3. Download Main Checkpoint
    checkpoint_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=cfg["checkpoint"],
        token=os.getenv("HF_TOKEN"),
        cache_dir=str(CACHE_DIR)
    )

    # 4. Download Vocoder (If Tacotron)
    vocoder_path = None
    if cfg.get("architecture") == "Tacotron2":
        vocoder_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=cfg["vocoder"],
            token=os.getenv("HF_TOKEN"),
            cache_dir=str(CACHE_DIR)
        )
        
        # NOTE: If your Tacotron class requires hifi-config.json, 
        # you must also download it here:
        hf_hub_download(repo_id=REPO_ID, filename="hifi-config.json", cache_dir=str(CACHE_DIR))

    # 5. Initialize and Load Weights
    # We use your existing class structure
    arch = cfg.get("architecture", "Tacotron2")
    cls = MODEL_CLASSES[arch]
    model_instance = cls()

    # Call your class's native .load() method with the HF paths
    if arch == "Tacotron2":
        model_instance.load(checkpoint_path, vocoder_path, **extra_settings)
    else:
        model_instance.load(checkpoint_path, **extra_settings)

    # 6. Cache the live object
    _loaded_models[model_name] = model_instance
    return model_instance


#########For Front End#################
def list_models():
    """Return dict of all available models from config.json"""
    return CONFIG_PATH

def list_speakers(model_name: str) -> list[str]:
    """Return list of speaker names for a specific model."""
    models = list_models()                  # returns the full registry dict
    model = models.get(model_name)
    if not model:
        return []

    # New registry style: speaker_ids is now a dict {"name": id}
    spk_dict = model.get("speaker_ids", {})
    

    # So like, the difrence is that now there are no extrnal files, it will be easier for HF, as I just need VilligerAnon/41.ai/model.pth or whatever
    # (The HF will be private just because I love you)
    return [f"{id}|{name}" for name, id in sorted(spk_dict.items(), key=lambda x: x[1])]
    

def list_info(model_name: str):
    """Return list of info for a specific model."""
    models = list_models()
    model = models.get(model_name)
    
    if not model:
        return []
    
    model_info = model.get("metadata") # Info used to be internal model info shit in the leagcy backend, nowadays it just displays info, hance the change from Info to metadata

    return [model_info] #Json bullshit

def list_tags(model_name: str):
    models = list_models()
    model = models.get(model_name)

    if not model:
        return []
    
    model_tags = model.get("tags")


    return model_tags
    