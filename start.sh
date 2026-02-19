#!/bin/bash
uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860}   # background FastAPI
wait   # keep container alive