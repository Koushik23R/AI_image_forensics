from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

BASE_DIR = Path(__file__).resolve().parents[1]
UPLOADS_DIR = BASE_DIR / "uploads"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
DB_DIR = BASE_DIR / "db"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

# In-memory storage kept intentionally close to the notebook design.
dataset_db: Dict[str, Dict[str, Any]] = {}
models_db: Dict[str, Dict[str, Any]] = {}
training_jobs_db: Dict[str, Dict[str, Any]] = {}


def _load_json_file(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            import json

            return json.loads(path.read_text())
    except Exception:
        return {}
    return {}


def _save_json_file(path: Path, data: Dict[str, Any]) -> None:
    try:
        import json

        path.write_text(json.dumps(data))
    except Exception:
        pass


# Load persisted DB state if available
_models_path = DB_DIR / "models.json"
_datasets_path = DB_DIR / "datasets.json"
_jobs_path = DB_DIR / "jobs.json"

models_db.update(_load_json_file(_models_path))
dataset_db.update(_load_json_file(_datasets_path))
training_jobs_db.update(_load_json_file(_jobs_path))


def generate_unique_id() -> str:
    return str(uuid.uuid4())


def save_model_metadata(model_id: str, metadata: Dict[str, Any]) -> None:
    models_db[model_id] = metadata
    _save_json_file(_models_path, models_db)


def get_model_metadata(model_id: str) -> Optional[Dict[str, Any]]:
    return models_db.get(model_id)


def save_dataset_metadata(dataset_id: str, metadata: Dict[str, Any]) -> None:
    dataset_db[dataset_id] = metadata
    _save_json_file(_datasets_path, dataset_db)


def get_dataset_metadata(dataset_id: str) -> Optional[Dict[str, Any]]:
    return dataset_db.get(dataset_id)


def create_training_job_record(job_id: str, dataset_id: str, epochs: int) -> Dict[str, Any]:
    record = {
        "job_id": job_id,
        "dataset_id": dataset_id,
        "status": "pending",
        "progress": 0,
        "epochs": epochs,
        "start_time": time.time(),
        "last_updated": time.time(),
        "model_id": None,
        "error": None,
        "metrics": {},
        "metrics_history": [],
        "logs": [],
        "current_epoch": 0,
    }
    training_jobs_db[job_id] = record
    return record


def update_training_job_status(
    job_id: str,
    status: str,
    progress: Optional[float] = None,
    metrics: Optional[Dict[str, Any]] = None,
    model_id: Optional[str] = None,
    error: Optional[str] = None,
    current_epoch: Optional[int] = None,
    log_message: Optional[str] = None,
) -> None:
    if job_id not in training_jobs_db:
        return

    job_record = training_jobs_db[job_id]
    job_record["status"] = status
    job_record["last_updated"] = time.time()

    if progress is not None:
        job_record["progress"] = progress
    if metrics is not None:
        job_record["metrics"] = metrics
        job_record.setdefault("metrics_history", []).append(
            {
                "epoch": current_epoch,
                **metrics,
            }
        )
    if model_id is not None:
        job_record["model_id"] = model_id
    if error is not None:
        job_record["error"] = error
    if current_epoch is not None:
        job_record["current_epoch"] = current_epoch
    if log_message:
        job_record.setdefault("logs", []).append(log_message)
    _save_json_file(_jobs_path, training_jobs_db)


def list_models() -> Dict[str, Dict[str, Any]]:
    return models_db


def list_datasets() -> Dict[str, Dict[str, Any]]:
    return dataset_db


def list_training_jobs() -> Dict[str, Dict[str, Any]]:
    return training_jobs_db
