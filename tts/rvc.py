from rvc_python.infer import RVCInference
from .utils import DEVICE
from .base import TTSModel
import os
from torch.serialization import add_safe_globals
from fairseq.data.dictionary import Dictionary

# Allow fairseq dictionary class to be unpickled
add_safe_globals([Dictionary])


class RVC(TTSModel):
    def __init__(self):
        self.model = None

    def load(self, model_path: str, vocoder_path: str = None, **extra_settings):
        """Load the RVC model checkpoint."""
        self.model = RVCInference(device=DEVICE)
        self.model.load_model(model_path_or_name=model_path, version="v2")
        print(f"Loaded RVC model from {model_path}")

    def synthesize(self, text=None, speaker=None, ref_audio=None, emotion=None, output_file=None, **extra_settings) -> str:
        if ref_audio is None:
            raise ValueError("RVC requires a reference audio file (`ref_audio`).")

        if output_file is None:
            os.makedirs("generated_audio", exist_ok=True)
            output_file = os.path.join(
                "generated_audio", "rvc_" + os.path.basename(ref_audio)
            )

        # Optional: pull extra_settings if provided
        pitch_shift = extra_settings.get("pitch_shift", 0)
        index_rate = extra_settings.get("index_rate", 1.0)
        filter_radius = extra_settings.get("filter_radius", 3)
        protect = extra_settings.get("protect", 0.33)
        extraction_method = extra_settings.get("extraction_method", 'harvest') # Choose between 'harvest', 'crepe', 'rmvpe', 'pm' (Would've wished if I addded more extraction methods later)

        #Add configuration from extra_settings if available
        self.model.f0method = extraction_method
        self.model.f0up_key = pitch_shift
        self.model.index_rate = index_rate
        self.model.filter_radius = filter_radius
        self.model.protect = protect #Forgot what this does
        self.model.infer_file(ref_audio, output_file)

        return output_file
