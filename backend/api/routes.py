from __future__ import annotations

import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

try:
    from backend.database.store import (
        UPLOADS_DIR,
        get_model_metadata,
        list_datasets,
        list_models,
        list_training_jobs,
    )
    from backend.services.statistics import get_statistics_for_model
    from backend.services.training import (
        generate_unique_id,
        get_training_job_status,
        handle_dataset_upload,
        perform_prediction,
        start_training_job,
    )
except ModuleNotFoundError:
    from database.store import (
        UPLOADS_DIR,
        get_model_metadata,
        list_datasets,
        list_models,
        list_training_jobs,
    )
    from services.statistics import get_statistics_for_model
    from services.training import (
        generate_unique_id,
        get_training_job_status,
        handle_dataset_upload,
        perform_prediction,
        start_training_job,
    )

router = APIRouter(prefix="/api")


class TrainRequest(BaseModel):
    dataset_id: str
    hyperparameters: dict | None = None


class PredictRequest(BaseModel):
    model_id: str
    image_path: str | None = None


@router.post("/dataset/upload")
async def upload_dataset(
    real_files: List[UploadFile] = File(default=[]),
    fake_files: List[UploadFile] = File(default=[]),
    files: List[UploadFile] = File(default=[]),
    labels: str = Form(default="[]"),
    generator_type: str = Form(...),
    img_size: str = Form("128,128"),
):
    try:
        parsed_img_size = tuple(int(value) for value in img_size.split(","))
        parsed_labels = json.loads(labels)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid form payload: {exc}") from exc

    dataset_id = generate_unique_id()
    dataset_dir = UPLOADS_DIR / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    real_dir = dataset_dir / "real"
    fake_dir = dataset_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    real_files_info = []
    fake_files_info = []

    if real_files or fake_files:
        for upload_file in real_files:
            destination = real_dir / Path(upload_file.filename).name
            content = await upload_file.read()
            destination.write_bytes(content)
            real_files_info.append({"path": str(destination), "filename": destination.name})

        for upload_file in fake_files:
            destination = fake_dir / Path(upload_file.filename).name
            content = await upload_file.read()
            destination.write_bytes(content)
            fake_files_info.append({"path": str(destination), "filename": destination.name})

        if not real_files_info:
            raise HTTPException(status_code=400, detail="Please upload a real folder with at least one image.")
        if not fake_files_info:
            raise HTTPException(status_code=400, detail="Please upload a fake folder with at least one image.")
    else:
        if len(parsed_labels) != len(files):
            raise HTTPException(status_code=400, detail="The number of labels must match the number of uploaded files.")

        for index, upload_file in enumerate(files):
            label_value = parsed_labels[index]
            label_name = "real" if label_value == 0 else "fake"
            class_dir = dataset_dir / label_name
            class_dir.mkdir(parents=True, exist_ok=True)

            destination = class_dir / Path(upload_file.filename).name
            content = await upload_file.read()
            destination.write_bytes(content)
            if label_value == 0:
                real_files_info.append({"path": str(destination), "filename": destination.name})
            else:
                fake_files_info.append({"path": str(destination), "filename": destination.name})

    metadata = handle_dataset_upload(
        real_files_info=real_files_info,
        fake_files_info=fake_files_info,
        generator_type=generator_type,
        img_size=parsed_img_size,
        dataset_id=dataset_id,
    )
    return metadata


@router.post("/train")
def train_model(request: TrainRequest):
    try:
        return start_training_job(request.dataset_id, request.hyperparameters)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/train/status/{job_id}")
def train_status(job_id: str):
    return get_training_job_status(job_id)


@router.post("/predict")
async def predict_image(
    model_id: str = Form(...),
    file: UploadFile = File(...),
):
    model_metadata = get_model_metadata(model_id)
    if not model_metadata:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found.")

    prediction_dir = UPLOADS_DIR / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    # Use only the uploaded file's base name to avoid nested paths
    from pathlib import Path as _Path

    safe_name = _Path(file.filename).name
    image_path = prediction_dir / safe_name
    image_path.write_bytes(await file.read())

    result = perform_prediction(str(image_path), model_id)
    result["image_path"] = str(image_path)
    return result


@router.get("/models")
def get_models():
    return {"models": list_models()}


@router.get("/datasets")
def get_datasets():
    return {"datasets": list_datasets()}


@router.get("/jobs")
def get_jobs():
    return {"jobs": list_training_jobs()}


@router.get("/statistics/{model_id}")
def get_statistics(model_id: str):
    try:
        return get_statistics_for_model(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
