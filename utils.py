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


from uberduck_ml_dev.text.util import convert_to_arpabet

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
