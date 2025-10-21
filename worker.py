# worker.py
from celery import Celery
from tts.registry import get_model
import os
app = Celery("tts_worker", broker="redis://localhost:6379/0")

@app.task
def synthesize_tts(job_id, text, model_name, speaker_id, emotion=None, **kwargs):
    # kwargs contains all the extra_settings
    extra_settings = kwargs
    
    print(f"Starting TTS job: {job_id}")
    print(f"Text: {text}")
    print(f"Model: {model_name}, Speaker: {speaker_id}")
    print(f"Emotion: {emotion}")
    print(f"Extra settings: {extra_settings}")
    
    model = get_model(model_name=model_name, extra_settings=extra_settings)
    output_file = os.path.join("generated_audio", f"{job_id}_gen.wav")
    # Call synthesize correctly - emotion should be passed as torchmoji_text or in extra_settings
    # Option 1: If you want to use emotion as torchmoji_text
    audio_path = model.synthesize(text, speaker_id, torchmoji_text=emotion, output_file=output_file ,**extra_settings)
    
    # Option 2: If you want to keep emotion separate from torchmoji_text
    # audio_path = model.synthesize(text, speaker_id, **extra_settings)
    
    return audio_path