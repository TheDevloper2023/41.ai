#This contains some functions used everywhere, so uhh yeah.

import sys
import torch

import json
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_CPU = DEVICE == "cpu"
sys.path.append("hifi-gan")
from env import AttrDict
from models import Generator
from denoiser import Denoiser

def load_hifigan(path, conf_name):
    conf = conf_name
    with open(conf) as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    hifigan = Generator(h).to(torch.device(DEVICE))
    state_dict_g = torch.load(path, map_location=torch.device(DEVICE))
    hifigan.load_state_dict(state_dict_g["generator"])
    hifigan.eval()
    hifigan.remove_weight_norm()
    denoiser = Denoiser(hifigan, mode="zeros")
    return hifigan, h, denoiser


from uberduck_ml_dev.text.util import convert_to_arpabet #Gota love the part where I have to update the whole thing from v2021 v2023 <3

def ARPAconverter(text: str) -> str:
    temp = convert_to_arpabet(text=text)
    # Strings are immutable; you must reassign
    temp = temp.replace("{ ", "{")
    temp = temp.replace(" }", "}")
    temp = temp.replace("{.}", ".")
    temp = temp.replace("{?}", "?")
    temp = temp.replace("{!}", "!")
    temp = temp.replace("} .", "}.")
    temp = temp.replace("} ,", "},")
    temp = temp.replace("} !", "}!")
    temp = temp.replace("} ?", "}?")
    temp = temp.replace("} '", "}'")
    return temp


import numpy as np
from skimage.transform import resize
def stretch_mel(mel, speaking_rate):
    """Stretch mel spectrogram by speaking rate."""
    print(f"DEBUG: mel shape: {mel.shape}")
    print(f"DEBUG: mel dimensions: {len(mel.shape)}")
    
    # Handle different input shapes
    if len(mel.shape) == 3:
        # Shape: [batch, n_mels, time]
        batch_size, n_mels, t = mel.shape
        print(f"DEBUG: 3D tensor - batch: {batch_size}, n_mels: {n_mels}, time: {t}")
        
        # Calculate new time length
        new_t = int(t / speaking_rate)
        
        # Convert to numpy and resize each batch
        mel_np = mel.detach().cpu().numpy() if isinstance(mel, torch.Tensor) else mel

        
        stretched_mels = []
        for i in range(batch_size):
            # Resize from [n_mels, t] to [n_mels, new_t]
            stretched_mel_batch = resize(mel_np[i], (n_mels, new_t))
            stretched_mels.append(stretched_mel_batch)
        
        # Stack back to 3D
        stretched_mel_np = np.stack(stretched_mels)
        
        # Convert back to tensor if input was tensor
        if isinstance(mel, torch.Tensor):
            stretched_mel = torch.FloatTensor(stretched_mel_np).to(mel.device)
        else:
            stretched_mel = stretched_mel_np
            
    elif len(mel.shape) == 2:
        # Shape: [n_mels, time]
        n_mels, t = mel.shape
        print(f"DEBUG: 2D tensor - n_mels: {n_mels}, time: {t}")
        
        # Calculate new time length
        new_t = int(t / speaking_rate)
        
        # Convert to numpy and resize
        mel_np = mel.detach().cpu().numpy() if isinstance(mel, torch.Tensor) else mel

        stretched_mel_np = resize(mel_np, (n_mels, new_t))
        
        # Convert back to tensor if input was tensor
        if isinstance(mel, torch.Tensor):
            stretched_mel = torch.FloatTensor(stretched_mel_np).to(mel.device)
        else:
            stretched_mel = stretched_mel_np
            
    else:
        raise ValueError(f"Unsupported mel spectrogram shape: {mel.shape}")
    
    print(f"DEBUG: stretched mel shape: {stretched_mel.shape}")
    return stretched_mel



MAX_WAV_VALUE = 32768.0