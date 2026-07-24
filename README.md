# AI Image Forensics

Local AI-generated image forensics application built from the notebook prototype.

## Project Structure

```text
project/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── api/
│   ├── ml/
│   ├── database/
│   ├── uploads/
│   └── saved_models/
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## Backend

The backend runs on FastAPI at http://localhost:8000.

Endpoints:

- `POST /api/dataset/upload`
- `POST /api/train`
- `GET /api/train/status/{job_id}`
- `POST /api/predict`
- `GET /api/models`
- `GET /api/datasets`
- `GET /api/jobs`
- `GET /api/statistics/{model_id}`

The backend preserves the notebook logic for:

- `dataset_db`
- `models_db`
- `training_jobs_db`
- FFT preprocessing
- Dual-Branch CNN construction
- rpy2 statistical analysis integration
- background training threads

## Frontend

The frontend runs on Vite at http://localhost:5173.

Pages:

- Dashboard
- Dataset Upload
- Train Model
- Training Monitor
- Prediction
- Statistics

## Run Locally

Backend:

```bash
cd backend
pip install -r requirements.txt
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Notes

- Uploaded files are stored under `backend/uploads/`.
- Trained model artifacts are stored under `backend/saved_models/`.
- The notebook-style in-memory state is intentionally preserved for datasets, models, and training jobs.
- R integration via `rpy2` is kept in the code path and works when a local R runtime is available, but the workspace install does not require it.