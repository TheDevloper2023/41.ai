# api.py
import uuid, os
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from worker import synthesize_tts

app = FastAPI()

@app.post("/tts")
def tts_request(
    text: str,
    model_name: str = "Tacotron2_English",
    speaker_id: int = 0,
    emotion: str = None
):
    job_id = str(uuid.uuid4())
    synthesize_tts.delay(job_id, text, model_name, speaker_id, emotion)
    return {"job_id": job_id, "model": model_name}

@app.get("/result/{job_id}")
def get_result(job_id: str):
    filepath = f"results/{job_id}.wav"
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="audio/wav")
    return JSONResponse({"status": "pending"})
