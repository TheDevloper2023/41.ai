from tts.registry import get_model
import os
import tempfile
#update: No more celery, it'll be easier to host

def synthesize_tts(job_id, text, model_name, speaker_id, emotion=None, ref_audio=None,**kwargs):
    # kwargs contains all the extra_settings
    extra_settings = kwargs
    
    print(f"Starting TTS job: {job_id}")
    print(f"Text: {text}")
    print(f"Model: {model_name}, Speaker: {speaker_id}")
    print(f"Emotion: {emotion}")
    print(f"Extra settings: {extra_settings}")
    
    model = get_model(model_name=model_name, extra_settings=extra_settings)
    output_file = os.path.join(tempfile.gettempdir(), f"{job_id}_gen.wav")
    audio_path = model.synthesize(text, speaker_id, torchmoji_text=emotion, output_file=output_file ,**extra_settings, ref_audio=ref_audio)
    return audio_path