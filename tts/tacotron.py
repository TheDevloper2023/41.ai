import os
import time
import torch
import torchaudio
import numpy as np
from scipy.signal import lfilter, firwin
from scipy.io.wavfile import write
from .base import TTSModel
from .utils import load_hifigan, ARPAconverter, DEVICE, USE_CPU, stretch_mel, MAX_WAV_VALUE
from uberduck_ml_dev.vendor.tfcompat.hparam import HParams
from uberduck_ml_dev.models.tacotron2 import Tacotron2, DEFAULTS
from uberduck_ml_dev.data_loader import prepare_input_sequence
from uberduck_ml_dev.models.torchmoji import TorchMojiInterface
import sys
sys.path.append("hifi-gan")
from meldataset import mel_spectrogram


class Tacotron2HIFI(TTSModel):
    def __init__(self):
        self.ttm = None  # Tacotron2 model
        self.mtw = None  # Hifi-GAN vocoder
        self.h = None    # Hifi-GAN config
        self.denoiser = None

        self.mtw_sr = None  # Super-resolution Hifi-GAN vocoder
        self.h2 = None
        self.denoiser_sr = None

        self.speaker_count = 1
        self.use_torchmoji = False
        self.torchmoji = None
        self.compute_gst = None

    def load(self, model_path: str, vocoder_path: str = None, **extra_settings):
        maxdecodesteps = extra_settings.get("maxdecodesteps", 3000)
        gate_threshold = extra_settings.get("gate_threshold", 0.05)

        # Load Tacotron2
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        config = DEFAULTS.values()

        # Speaker embedding
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

        # Torchmoji (GST)
        if "gst_lin.weight" in checkpoint:
            self.use_torchmoji = True
            self.torchmoji = TorchMojiInterface("vocabulary.json", "pytorch_model.bin")
            self.compute_gst = lambda texts: self.torchmoji.encode_texts(texts)
            config.update({
                "gst_dim": 2304,
                "gst_type": "torchmoji",
                "torchmoji_vocabulary_file": "vocabulary.json",
                "torchmoji_model_file": "pytorch_model.bin",
            })
        else:
            self.use_torchmoji = False

        # Config updates
        config.update({
            "max_decoder_steps": maxdecodesteps,
            "symbol_set": "nvidia_taco2",
            "text_cleaners": ["english_cleaners"],
            "gate_threshold": gate_threshold,
            "p_attention_dropout": 0.0,
            "p_decoder_dropout": 0.0,
            "num_mels": 100,
            "n_fft": 2048,
        })

        hparams = HParams(**config)
        self.ttm = Tacotron2(hparams)
        self.ttm.from_pretrained(model_path, DEVICE)

        # Custom inference wrapper
        tacotron = self.ttm
        @torch.no_grad()
        def custom_inference(inputs):
            text, input_lengths, speaker_ids, embedded_gst, *_ = inputs
            embedded_inputs = tacotron.embedding(text).transpose(1, 2)
            embedded_text = tacotron.encoder.inference(embedded_inputs, input_lengths)
            encoder_outputs = embedded_text

            if tacotron.speaker_embedding:
                speakers = torch.arange(tacotron.n_speakers, device=DEVICE)
                embeddings = tacotron.speaker_embedding(speakers.unsqueeze(0))[0]
                average = torch.mean(embeddings, 0)
                embedding_offsets = embeddings - average
                mixed_offset = torch.sum(embedding_offsets * speaker_ids[:, None], 0)
                embedded_speakers = mixed_offset + average
                encoder_outputs += tacotron.spkr_lin(embedded_speakers)

            if tacotron.gst_lin is not None:
                assert embedded_gst is not None
                gst_embedding = tacotron.gst_lin(embedded_gst)
                style_weight = 1.0 + 0.7 * embedded_gst.mean()
                encoder_outputs += style_weight * gst_embedding

            memory_lengths = input_lengths
            mel_outputs, gate_outputs, alignments, mel_lengths = tacotron.decoder.inference(
                encoder_outputs, memory_lengths
            )
            mel_outputs_postnet = tacotron.postnet(mel_outputs)
            mel_outputs_postnet = mel_outputs + mel_outputs_postnet

            return tacotron.parse_output(
                [mel_outputs, mel_outputs_postnet, gate_outputs, alignments, mel_lengths]
            )

        self.ttm.inference = custom_inference
        vocoder_path = "g_02500000" #Patch for HF for now
        # Load Hifi-GAN vocoder
        self.mtw, self.h, self.denoiser = load_hifigan(vocoder_path, os.path.join(os.path.dirname(vocoder_path), "hifi-config.json"))
        # Load super-resolution Hifi-GAN
        self.mtw_sr, self.h2, self.denoiser_sr = load_hifigan("Superres_Twilight_33000", "config_32k.json")


        print("Models loaded successfully")

    def synthesize(self, text, speaker_id, torchmoji_text=None, output_file="" ,**extra_settings) -> str:
        if self.h2 is None:
            raise RuntimeError("Super-resolution model not loaded. Call load() first.")

        skip_sr = extra_settings.get("skip_sr", False)
        superress = extra_settings.get("superress", 0.5)  # safer default
        arpaconv = extra_settings.get("arpaconv", True)
        denoise1 = extra_settings.get("denoise", 35)
        speaking_rate = extra_settings.get("speaking_rate", 1.2) #Will be added when the backend isn't shit (like my life)

        torchmoji_text = torchmoji_text if torchmoji_text is not None else text
        if arpaconv:
            text = ARPAconverter(text)

        # Compute GST embedding
        if self.use_torchmoji:
            with torch.inference_mode():
                embedding = torch.FloatTensor(self.compute_gst([torchmoji_text]))
        else:
            embedding = torch.zeros(1, 1, 200)

        if not USE_CPU:
            embedding = embedding.to(DEVICE)

        # Speaker embedding
        speaker_embedding = torch.zeros(self.speaker_count, device=DEVICE)
        speaker_embedding[speaker_id] = 1

        # Prepare input sequence
        text_padded, input_lengths = prepare_input_sequence(
            [text], cpu_run=USE_CPU, arpabet=1.0, symbol_set="nvidia_taco2"
        )
        if not USE_CPU:
            text_padded = text_padded.to(DEVICE)
            input_lengths = input_lengths.to(DEVICE)

        input_ = [text_padded, input_lengths, speaker_embedding, embedding]

        # Tacotron2 inference
        with torch.no_grad(), torch.inference_mode():
            tacotron_output = self.ttm.inference(input_)
            mel_postnet = tacotron_output[1][:1]  # mel spectrogram

        def vocode_mel_spectrogram(mel_postnet):
                """Vocode mel spectrogram using HiFi-GAN with denoising."""
                with torch.inference_mode():
                    # HiFi-GAN forward
                    print("Vocode")
                    audio = self.mtw(mel_postnet)  # [1, T]
                    audio = audio.squeeze()
                    audio *= MAX_WAV_VALUE

            
                    # Denoise (still torch)
                    print("Denoise")
                    audio_denoised = self.denoiser(audio.view(1, -1), strength=denoise1)[:, 0]

                    # Convert to numpy
                    print("Convert numpy")
                    audio_denoised = audio_denoised.cpu().detach().numpy().reshape(-1)

                return audio_denoised.astype(np.float32)
        def resample_audio(audio, original_sr, target_sr):
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu()
                else:
                    audio = torch.from_numpy(audio).float()

                if len(audio.shape) == 1:
                    audio = audio.unsqueeze(0)

                resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
                audio_resampled = resampler(audio)
                return audio_resampled.squeeze(0).numpy()

        def apply_super_resolution(base_audio):
                print("SR")
                """Apply HiFi-GAN super-resolution with high-pass filtering."""
                wave = base_audio.astype(np.float32) / MAX_WAV_VALUE
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

                sr_hat = self.mtw_sr(mel).squeeze() * MAX_WAV_VALUE
                sr_audio = sr_hat.detach().cpu().numpy().reshape(-1)
                sr_audio = sr_audio.astype(np.float32)
                
                # Apply high-pass filter
                hp_b = firwin(401, cutoff=10500, fs=self.h2.sampling_rate, pass_zero=False)
                sr_audio = lfilter(hp_b, 1.0, sr_audio) * 1.2
                high_freqs = float(superress) * sr_audio
                return high_freqs.astype(np.float32)

        def merge_audio(original, superres, normalize=False):
                """Merge base and super-resolution audio."""
                min_len = min(len(original), len(superres))
                original = original[:min_len].astype(np.float32)
                superres = superres[:min_len].astype(np.float32)

                merged = original + superres

                if normalize:
                    peak = np.max(np.abs(merged))
                    if peak > 0:
                        merged *= 0.98 / peak

                merged = np.clip(merged, -1, 1)
                final_audio_int16 = (merged * 32767).astype(np.int16)
                return final_audio_int16
        

        audio_denoised = vocode_mel_spectrogram(mel_postnet)
        if skip_sr:
                # Skip super-resolution, just use base audio
                audio_final = audio_denoised
                audio_final = audio_final / np.max(np.abs(audio_final))
                sample_rate = self.h.sampling_rate
        else:
                # Apply super-resolution
                sample_rate = self.h2.sampling_rate
                audio_denoised = resample_audio(audio_denoised, self.h.sampling_rate, self.h2.sampling_rate)
                audio2_denoised = apply_super_resolution(audio_denoised)
                audio_final = merge_audio(audio_denoised, audio2_denoised, normalize=True)

        audio_pth = output_file
        write(audio_pth, sample_rate, audio_final)
            
        return audio_pth
