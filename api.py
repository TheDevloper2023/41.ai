import uuid
import os
from typing import Optional
import tempfile
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import BackgroundTasks
from worker import synthesize_tts
from tts.registry import list_models, list_speakers, last_used, CACHE_DIR, _loaded_models as model_cache
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import asyncio
import shutil
from pathlib import Path

app = FastAPI()
templates = Jinja2Templates(directory="templates")
temp_dir = tempfile.gettempdir()

CACHE_TIMEOUT_MIN = 15    
CACHE_DIR = "/tmp/hf_cache"

# So uhh, on HF; you get a few gigs for free. and if I download all models it will fill up
# To prevend this we download each time, wait a period and unused models in that period will be deleted from the space kinda

# I was lazy so this was written by AI, I just want to return the site already.
async def cleanup_idle_models():
    """Runs every few minutes — deletes old models"""
    while True:
        now = datetime.utcnow()
        to_delete = []

        for name, ts in last_used.items():
            if now - ts > timedelta(minutes=CACHE_TIMEOUT_MIN):
                to_delete.append(name)

        for name in to_delete:
            if name in model_cache:
                del model_cache[name]
            # Also delete from disk if you want (optional but recommended)
            model_folder = Path(CACHE_DIR) / name
            if model_folder.exists():
                shutil.rmtree(model_folder, ignore_errors=True)
            del last_used[name]
            print(f"Expired and deleted model: {name}")

        await asyncio.sleep(300)  # check every 5 minutes

        

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_idle_models())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/tts")
async def tts_request(
    background_tasks: BackgroundTasks,
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

    ##### RVC #####

    pitch_shift: int = Form(0),
    index_rate: float = Form(1.0),
    filter_radius: int = Form(3),
    extraction_method: str = Form("harvest"),
    protect: float = Form(.33)
):
    job_id = str(uuid.uuid4())

    ref_audio_path = None
    if ref_audio:
        os.makedirs(temp_dir = tempfile.gettempdir(), exist_ok=True)
        ref_audio_path = os.path.join(temp_dir, f"{job_id}_{ref_audio.filename}")
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

        ### RVC ###
        "pitch_shift": pitch_shift,
        "index_rate": index_rate,
        "filter_radius": filter_radius,
        "extraction_method": extraction_method,
        "protect": protect,
    }

    # enqueue TTS job
    #synthesize_tts.delay(job_id, text, model_name, speaker_id, emotion, **extra_settings, ref_audio=ref_audio_path)
    background_tasks.add_task(synthesize_tts, job_id, text, model_name, speaker_id, emotion, **extra_settings, ref_audio=ref_audio_path)

    return {
        "job_id": job_id,
        "model": model_name,
        "extra_settings": extra_settings,
        "ref_audio": ref_audio_path,
    }


@app.get("/result/{job_id}")
def get_result(job_id: str):
    output_file = f"{tempfile.gettempdir()}/{job_id}_gen.wav"
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