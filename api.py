import uuid
import os
from typing import Optional

from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from worker import synthesize_tts
from tts.registry import list_models, list_speakers

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.post("/tts")
async def tts_request(
    request: Request,
    text: str = Form(...),
    model_name: str = Form("Tacotron2_English"),
    speaker_id: int = Form(0),
    emotion: Optional[str] = Form(None),
    ref_audio: Optional[UploadFile] = File(None),
    maxdecodesteps: int = Form(3000),
    gate_threshold: float = Form(0.05),
    speaking_rate: float = Form(1.2),
    superress: float = Form(4.0),
    denoise: int = Form(50),
    skip_sr: bool = Form(False),
    arpaconv: bool = Form(True),
):
    job_id = str(uuid.uuid4())

    ref_audio_path = None
    if ref_audio:
        os.makedirs("uploaded_refs", exist_ok=True)
        ref_audio_path = os.path.join("uploaded_refs", f"{job_id}_{ref_audio.filename}")
        with open(ref_audio_path, "wb") as f:
            f.write(await ref_audio.read())

    extra_settings = {
        "maxdecodesteps": maxdecodesteps,
        "gate_threshold": gate_threshold,
        "speaking_rate": speaking_rate,
        "superress": superress,
        "denoise": denoise,
        "skip_sr": skip_sr,
        "arpaconv": arpaconv,
    }

    # enqueue TTS job
    synthesize_tts.delay(job_id, text, model_name, speaker_id, emotion, **extra_settings, ref_audio=ref_audio_path)

    return {
        "job_id": job_id,
        "model": model_name,
        "extra_settings": extra_settings,
        "ref_audio": ref_audio_path,
    }


@app.get("/result/{job_id}")
def get_result(job_id: str):
    output_file = f"generated_audio/{job_id}_gen.wav"
    if os.path.exists(output_file):
        return FileResponse(output_file, media_type="audio/wav")
    return JSONResponse({"status": "pending"})


@app.get("/", response_class=HTMLResponse)
def main(request: Request):
    models = list_models()
    return templates.TemplateResponse("index.html", {"request": request, "models": models})


@app.get("/speakers/{model_name}")
async def get_speakers(model_name: str):
    """Return available speakers for a given model."""
    speakers = list_speakers(model_name)
    return {"model": model_name, "speakers": speakers}


@app.get("/favicon.ico")
def favicon():
    return FileResponse("/templates/favicon.ico")
