from .base import TTSModel
from .base import DEVICE, USE_CPU
from ..utils import load_hifigan, ARPAconverter
from uberduck_ml_dev.vendor.tfcompat.hparam import HParams
from uberduck_ml_dev.models.tacotron2 import Tacotron2, DEFAULTS
from uberduck_ml_dev.data_loader import prepare_input_sequence
from uberduck_ml_dev.models.torchmoji import TorchMojiInterface
import sys
sys.path.append("..hifi-gan")
from meldataset import mel_spectrogram
from scipy.signal import lfilter, firwin
import torch
import torchaudio
import os
import numpy as np
import time

class Tacotron2(TTSModel):
    def __init__(self):
        self.ttm = None #Tacotron2 Model
        self.mtw = None #Hifi-gan model
    def load(self, model_path: str):
        #Load TT2
        checkpoint = torch.load(model_path + "/taco.pt", map_location=DEVICE, weights_only=False)

        if "model" in checkpoint.keys(): checkpoint = checkpoint["model"] 
        if "state_dict" in checkpoint.keys(): checkpoint = checkpoint["state_dict"]
        config = DEFAULTS.values()
        #Load SpkEmb

        if "speaker_embedding.weight" in checkpoint:
            self.speaker_count = len(checkpoint["speaker_embedding.weight"])
            config.update({
                    "has_speaker_embedding": True,
                    "n_speakers": self.speaker_count,
                    "ignore_layers": ["null"],
                })
            print(f"{self.speaker_count} speakers found in model")
        else:
            self.speaker_count = 1
            print("No speaker embedding found in model, defaulting to single speaker")
        
        #Load Torchmoji

        if "gst_lin.weight" in checkpoint:
            self.use_torchmoji = True
            self.torchmoji = TorchMojiInterface(
                "vocabulary.json",
                "pytorch_model.bin",
            )
            self.compute_gst = lambda texts: self.torchmoji.encode_texts(texts)
            config.update({
                "gst_dim": 2304,
                "gst_type": "torchmoji",
                "torchmoji_vocabulary_file": "vocabulary.json",
                "torchmoji_model_file": "pytorch_model.bin",
            })
        else:
            self.use_torchmoji = False


        # Update configurations
        config.update({
                ### Basic Things ###
                "max_decoder_steps": 3000,  # 30 secs audio
                "symbol_set": "nvidia_taco2",
                "text_cleaners": ["english_cleaners"],
                "gate_threshold": 0.05, # Hopefuly prevent strokes (Mostly useless)

                
                ### Some tricks to make the model sound better ###
                "p_attention_dropout": 0.00,
                "p_decoder_dropout": 0.00,
                "num_mels": 100,
                "n_fft":2048
            })
        

        hparams = HParams(**config)
        self.ttm = Tacotron2(hparams)
        self.ttm.from_pretrained(model_path + "/taco.pt", device=DEVICE)
        tacotron = self.ttm
        @torch.no_grad()
        def custom_inference(inputs):
                """Custom inference method for Tacotron2."""
                text, input_lengths, speaker_ids, embedded_gst, *_ = inputs
                
                # Text embedding - use tacotron.embedding instead of self.embedding
                embedded_inputs = tacotron.embedding(text).transpose(1, 2)
                embedded_text = tacotron.encoder.inference(embedded_inputs, input_lengths)
                encoder_outputs = embedded_text
                
                # Speaker embedding
                if tacotron.speaker_embedding:
                    speakers = torch.arange(tacotron.n_speakers, device=DEVICE)
                    embeddings = tacotron.speaker_embedding(speakers.unsqueeze(0))[0]
                    average = torch.mean(embeddings, 0)
                    embedding_offsets = embeddings - average
                    mixed_offset = torch.sum(embedding_offsets * speaker_ids[:, None], 0)
                    embedded_speakers = mixed_offset + average
                    encoder_outputs += tacotron.spkr_lin(embedded_speakers)
                
                # Style embedding (GST)
                if tacotron.gst_lin is not None:
                    assert (
                        embedded_gst is not None
                    ), f"embedded_gst is None but gst_type was set to {tacotron.gst_type}"
                    gst_embedding = tacotron.gst_lin(embedded_gst) 
                    style_weight = 1.0 + 0.7 * embedded_gst.mean()
                    encoder_outputs += style_weight * gst_embedding
                    
                # Decoder
                memory_lengths = input_lengths
                mel_outputs, gate_outputs, alignments, mel_lengths = tacotron.decoder.inference(
                    encoder_outputs, memory_lengths
                )
                mel_outputs_postnet = tacotron.postnet(mel_outputs)
                mel_outputs_postnet = mel_outputs + mel_outputs_postnet * 1.7
                
                return tacotron.parse_output(
                    [mel_outputs, mel_outputs_postnet, gate_outputs, alignments, mel_lengths]
                )
        
        self.ttm.inference = custom_inference

        # Load Hifi-Gan
        self.mtw, self.h ,self.denoiser = load_hifigan(model_path + "/vocoder", model_path + "/hifi_config.json")
        self.mtw_sr, self.h2 ,self.denoiser_sr = load_hifigan("Superres_Twilight_33000", "config_32k.json")

        print("Models loaded successfully")
    def synthesize(self, text, speaker_id, torchmoji_text=None, superress=4, arpaconv=True, skip_sr=False) -> bytes:
        #Convert T2Arpa
        if arpaconv:
            text = ARPAconverter(text)
        
        #Precompute Embbedings
        if self.use_torchmoji:
            with torch.inference_mode():
                embedding = self.compute_gst([torchmoji_text])
                embedding = torch.FloatTensor(embedding)
                #emojis = self.torchmoji.enc2emojis(embedding)[0]
        else:
            embedding = torch.zeros(1, 1, 200)
                #emojis = []

        if not USE_CPU:
            embedding = embedding.to(DEVICE)
        
        #Spekaer Embdding
        speaker_embedding = [0] * self.speaker_count
        speaker_embedding[speaker_id] = 1
        speaker_embedding = torch.FloatTensor(speaker_embedding)
        if not USE_CPU:
            speaker_embedding = speaker_embedding.to(DEVICE)
        
        #Preper input_seq
        text_padded, input_lengths = prepare_input_sequence(
                [text], cpu_run=USE_CPU, arpabet=1.0, symbol_set="nvidia_taco2"
        )

        if not USE_CPU:
            text_padded = text_padded.to(DEVICE)
            input_lengths = input_lengths.to(DEVICE)

        input_ = [text_padded, input_lengths, speaker_embedding, embedding]

        #Run Tacotron
        with torch.inference_mode(), torch.no_grad():
                input = self.tacotron.inference(input_) # shape: [batch, n_mel, time]
                output = input[1][:1] #This is the mel spectrogram

        #Run Vocoder

        #Vocode
        with torch.inference_mode():
            audio = self.mtw(output)
            audio = audio.squeeze()
            audio *= 32768.0 #MAX_WAV_VAL

        #Denoise
            audio_denoised = self.denoiser(audio.view(1, -1), strength=50)[:, 0]
            audio_denoised = audio_denoised.cpu().detach().numpy().reshape(-1)
            audio_denoised.astype(np.float32)

        #Resample
        if isinstance(audio_denoised, torch.Tensor):
            audio_denoised = audio_denoised.detach().cpu()
        else:
            audio_denoised = torch.from_numpy(audio).float()

        if len(audio_denoised.shape) == 1:
            audio_denoised = audio.unsqueeze(0)
        
        resampler = torchaudio.transforms.Resample(orig_freq=self.h.sampling_rate, new_freq=self.h2.sampling_rate)
        audio_denoised = resampler(audio_denoised)
        audio_denoised.squeeze(0).numpy()

        #Superres
        wave = audio_denoised.astype(np.float32) / 32768.0
        wave = torch.FloatTensor(wave).to(DEVICE)
        mel = mel_spectrogram(
                    wave.unsqueeze(0),
                    self.h2.n_fft,
                    self.h2.num_mels,
                    self.h2.sampling_rate,
                    self.h2.hop_size,
                    self.h2.win_size,
                    self.h2.fmin,
                    self.h2.fmax,
                )

        sr_hat = self.hifigan_sr(mel).squeeze() * 32768.0
        sr_audio = sr_hat.detach().cpu().numpy().reshape(-1)
        sr_audio = sr_audio.astype(np.float32)
        hp_b = firwin(401, cutoff=10500, fs=self.h2.sampling_rate, pass_zero=False)
        sr_audio = lfilter(hp_b, 1.0, sr_audio) * 1.2
        audio_denoised2 = float(superress) * sr_audio
        audio_denoised2 = audio_denoised2.astype(np.float32)

        #Merge
        min_len = min(len(audio_denoised), len(audio_denoised2))
        audio_denoised = audio_denoised[:min_len].astype(np.float32)
        audio_denoised2 = audio_denoised2[:min_len].astype(np.float32)

        merged = audio_denoised + audio_denoised2

        #Normalize
        peak = np.max(np.abs(merged))
        if peak > 0:
            merged *= 0.98 / peak

        merged = np.clip(merged, -1, 1)
        final_audio_int16 = (merged * 32767).astype(np.int16) #This is the audio the user gets
        
        #Convert the .wav and return it's path
        audio_pth = os.path.join("generated_audio", f"{int(time.time() * 1000)}_tacotron_gen.wav")
        from scipy.io.wavfile import write
        write(audio_pth, self.h2.sampling_rate ,final_audio_int16)

        return audio_pth
        