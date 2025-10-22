import torch
from collections import OrderedDict
model = torch.load("taco.pt", "cpu")
model_finalized = OrderedDict()
model_finalized["state_dict"] = model["model"]
torch.save(model_finalized, "taco.pt")