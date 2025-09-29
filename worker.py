# worker.py
from celery import Celery
from tts.registry import get_model
import os

app = Celery("tts_worker", broker="redis://localhost:6379/0")

@app.task
def synthesize_tts(job_id, text, model_name="Tacotron2_English", speaker_id=0, emotion=None):
    model = get_model(model_name)
    audio_path = model.synthesize(text, speaker_id, emotion)

    os.makedirs("results", exist_ok=True)
    final_path = f"results/{job_id}.wav"
    os.rename(audio_path, final_path)

    return final_path
