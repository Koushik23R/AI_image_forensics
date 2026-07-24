from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

try:
    from backend.api.routes import router
    from backend.database.store import SAVED_MODELS_DIR, UPLOADS_DIR
except ModuleNotFoundError:
    from api.routes import router
    from database.store import SAVED_MODELS_DIR, UPLOADS_DIR

app = FastAPI(title="AI Image Forensics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/saved_models", StaticFiles(directory=str(SAVED_MODELS_DIR)), name="saved_models")


@app.get("/")
def health_check():
    return {"status": "ok", "message": "AI Image Forensics API is running."}
