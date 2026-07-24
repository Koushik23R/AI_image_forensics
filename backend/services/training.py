from __future__ import annotations

import json
import os
import shutil
import time
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

try:
    from backend.database.store import (
        SAVED_MODELS_DIR,
        UPLOADS_DIR,
        create_training_job_record,
        generate_unique_id,
        get_dataset_metadata,
        get_model_metadata,
        save_dataset_metadata,
        save_model_metadata,
        update_training_job_status,
    )
    from backend.ml.data import CustomDataGenerator
    from backend.ml.model import build_dual_branch_cnn
    from backend.ml.preprocessing import apply_fft_to_image, preprocess_image_for_spatial_branch
except ModuleNotFoundError:
    from database.store import (
        SAVED_MODELS_DIR,
        UPLOADS_DIR,
        create_training_job_record,
        generate_unique_id,
        get_dataset_metadata,
        get_model_metadata,
        save_dataset_metadata,
        save_model_metadata,
        update_training_job_status,
    )
    from ml.data import CustomDataGenerator
    from ml.model import build_dual_branch_cnn
    from ml.preprocessing import apply_fft_to_image, preprocess_image_for_spatial_branch

image_dimension = 128
model = build_dual_branch_cnn(input_shape=(image_dimension, image_dimension, 3))


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    return value


def handle_dataset_upload(
    image_files_info=None,
    labels=None,
    generator_type=None,
    img_size=(128, 128),
    dataset_id=None,
    real_files_info=None,
    fake_files_info=None,
):
    """
    Simulates processing an uploaded dataset.

    Args:
        image_files_info (list): Legacy list of dicts, each with 'path' and 'filename'.
        labels (list): Legacy corresponding labels (0 for real, 1 for fake).
        generator_type (str): E.g., 'Stable Diffusion', 'Real', 'GAN'.
        img_size (tuple): Target image size for preprocessing.
        real_files_info (list): Files uploaded from the real folder.
        fake_files_info (list): Files uploaded from the fake folder.

    Returns:
        dict: Dataset metadata.
    """
    dataset_id = dataset_id or generate_unique_id()
    timestamp = time.time()

    combined_files = []
    combined_labels = []

    if real_files_info is not None or fake_files_info is not None:
        for file_info in real_files_info or []:
            combined_files.append({**file_info, "class_name": "real", "label": 0})
            combined_labels.append(0)
        for file_info in fake_files_info or []:
            combined_files.append({**file_info, "class_name": "fake", "label": 1})
            combined_labels.append(1)
    else:
        combined_files = image_files_info or []
        combined_labels = labels or []
        if combined_files and not combined_labels:
            raise ValueError("Labels are required when using the legacy upload format.")

    metadata = {
        "dataset_id": dataset_id,
        "name": f"Dataset-{dataset_id[:8]}",
        "image_count": len(combined_files),
        "real_count": sum(1 for item in combined_files if item.get("label") == 0),
        "fake_count": sum(1 for item in combined_files if item.get("label") == 1),
        "labels": combined_labels,
        "generator_type": generator_type,
        "img_size": img_size,
        "uploaded_at": timestamp,
        "status": "processed",
        "files": combined_files,
    }
    save_dataset_metadata(dataset_id, metadata)
    return metadata


def _build_generators(dataset_metadata, img_size, batch_size=2):
    file_entries = dataset_metadata.get("files", [])
    all_image_paths = [entry["path"] for entry in file_entries]
    all_labels = dataset_metadata.get("labels", [])

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_image_paths,
        all_labels,
        test_size=0.3,
        random_state=42,
        stratify=all_labels,
    )
    train_generator = CustomDataGenerator(
        train_paths,
        train_labels,
        img_size=img_size,
        batch_size=batch_size,
    )
    val_generator = CustomDataGenerator(
        val_paths,
        val_labels,
        img_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_generator, val_generator


def _simulate_training_task(job_id, dataset_id, epochs):
    """A mock training task that updates its status periodically."""
    update_training_job_status(job_id, "running", progress=0, current_epoch=0, log_message="Training started")

    try:
        dataset_metadata = get_dataset_metadata(dataset_id)
        if not dataset_metadata:
            raise ValueError(f"Dataset {dataset_id} not found.")

        img_size = tuple(dataset_metadata.get("img_size", (image_dimension, image_dimension)))
        train_generator, val_generator = _build_generators(dataset_metadata, img_size=img_size, batch_size=2)

        for epoch in range(epochs):
            progress = ((epoch + 1) / epochs) * 100
            time.sleep(1)

            mock_loss = 1.0 - (progress / 100 * 0.8)
            mock_accuracy = 0.5 + (progress / 100 * 0.4)
            mock_val_loss = 1.0 - (progress / 100 * 0.7)
            mock_val_accuracy = 0.5 + (progress / 100 * 0.3)

            metrics = {
                "loss": mock_loss,
                "accuracy": mock_accuracy,
                "val_loss": mock_val_loss,
                "val_accuracy": mock_val_accuracy,
            }
            update_training_job_status(
                job_id,
                "running",
                progress=progress,
                metrics=metrics,
                current_epoch=epoch + 1,
                log_message=f"Epoch {epoch + 1}/{epochs} completed",
            )

        final_model_id = generate_unique_id()
        model_path = SAVED_MODELS_DIR / f"{final_model_id}.keras"
        mock_final_metrics = {"accuracy": 0.9, "precision": 0.85, "recall": 0.92, "f1": 0.88}
        save_model_metadata(
            final_model_id,
            {
                "model_id": final_model_id,
                "dataset_id": dataset_id,
                "training_job_id": job_id,
                "hyperparameters": {"epochs": epochs, "optimizer": "adam"},
                "metrics": mock_final_metrics,
                "saved_path": str(model_path),
                "trained_at": time.time(),
                "dataset_metadata": dataset_metadata,
            },
        )
        model.save(model_path)
        update_training_job_status(
            job_id,
            "completed",
            progress=100,
            metrics=mock_final_metrics,
            model_id=final_model_id,
            current_epoch=epochs,
            log_message="Training completed",
        )
    except Exception as e:
        update_training_job_status(job_id, "failed", error=str(e), log_message=f"Training failed: {e}")


def start_training_job(dataset_id, hyperparameters=None):
    """Initiates a training job as a background task."""
    if dataset_id not in get_all_dataset_ids():
        raise ValueError(f"Dataset {dataset_id} not found.")

    job_id = generate_unique_id()
    epochs = hyperparameters.get("epochs", 5) if hyperparameters else 5
    create_training_job_record(job_id, dataset_id, epochs)

    training_thread = Thread(target=_simulate_training_task, args=(job_id, dataset_id, epochs), daemon=True)
    training_thread.start()

    return {"job_id": job_id, "status": "pending"}


def get_training_job_status(job_id):
    return _safe_job_status(job_id)


def get_all_dataset_ids() -> List[str]:
    from backend.database.store import dataset_db

    return list(dataset_db.keys())


def _safe_job_status(job_id: str) -> Dict[str, Any]:
    from backend.database.store import training_jobs_db

    return training_jobs_db.get(job_id, {"job_id": job_id, "status": "not_found"})


def perform_prediction(image_path, model_id):
    """
    Simulates performing a prediction on a single image using a trained model.
    """
    model_metadata = get_model_metadata(model_id)
    if not model_metadata:
        return {"status": "error", "message": f"Model {model_id} not found."}

    global model
    if model is None:
        model = build_dual_branch_cnn(input_shape=(image_dimension, image_dimension, 3))

    try:
        spatial_img = preprocess_image_for_spatial_branch(image_path, (image_dimension, image_dimension))
        fft_img = apply_fft_to_image(image_path, (image_dimension, image_dimension))

        spatial_img_batch = np.expand_dims(spatial_img, axis=0)
        fft_img_batch = np.expand_dims(fft_img, axis=0)

        confidence = np.random.rand()
        predicted_class = 1 if confidence > 0.5 else 0
        probability = float(confidence)

        result = {
            "status": "success",
            "prediction": "AI-Generated" if predicted_class == 1 else "Real",
            "confidence": probability,
            "probability": probability,
            "fft_visualization": fft_img[:, :, 0].tolist(),
            "model_id": model_id,
        }
        return _to_json_safe(result)
    except FileNotFoundError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": f"Prediction failed: {e}"}
