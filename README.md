# Cats vs Dogs MLOps Pipeline

## Project Overview
End-to-end MLOps pipeline for binary image classification (Cats vs Dogs) for a pet adoption platform. Includes data versioning, model training, experiment tracking, packaging, containerization, and CI/CD setup.

## Folder Structure
- `src/` — Python scripts (preprocessing, training, inference API)
- `notebooks/` — Jupyter notebooks
- `data/` — Raw and processed datasets (DVC tracked)
- `tests/` — Unit tests
- `configs/` — Config files (DVC, MLflow, deployment)
- `.github/workflows/` — CI/CD pipeline
- `deployment/` — Deployment manifests
- `models/` — Saved models

## Steps Completed

### 1. Project Initialization
- Created recommended folder structure
- Initialized Git and DVC
- Added `.gitignore` for Python, DVC, and environment files

### 2. Data Versioning
- Downloaded Cats vs Dogs dataset from Kaggle
- Placed raw data in `data/raw/training_set/cats`, `data/raw/training_set/dogs`, `data/raw/test_set/cats`, `data/raw/test_set/dogs`
- Tracked raw and processed data with DVC
- Resolved DVC file ignore issue by updating `.gitignore`

### 3. Data Preprocessing
- Created `src/preprocess.py` to resize images to 224x224, split into train/val/test, and apply augmentation
- Updated script to match dataset folder structure
- Added required dependencies to `requirements.txt`

### 4. Model Development & Experiment Tracking
- Added PyTorch, torchvision, and MLflow to `requirements.txt`
- Created `src/train.py` for baseline CNN model
- Integrated MLflow for experiment tracking
- Trained model and logged metrics/artifacts
- Evaluated model on test set with `src/evaluate.py`

### 5. Inference API
- Added FastAPI and Uvicorn to `requirements.txt`
- Created `src/app.py` for REST API with `/health` and `/predict` endpoints

## Issues Faced & Solutions
- **DVC file ignored by git:**
  - Error: `bad DVC file name 'data\processed.dvc' is git-ignored.`
  - Solution: Removed `*.dvc` and `/data/processed/` from `.gitignore`.
- **Dependency conflicts (packaging, wheel, mlflow):**
  - Error: `mlflow 2.9.2 requires packaging <24, but you have packaging 26.0 which is incompatible.`
  - Solution: Downgraded `packaging` and `wheel` to compatible versions.
- **Wrong Python interpreter used:**
  - Error: `ModuleNotFoundError: No module named 'encodings'` (MySQL Shell Python)
  - Solution: Activated correct conda environment (`mlops-a2`).
- **MLflow warning about pkg_resources:**
  - Warning: `pkg_resources is deprecated as an API.`
  - Solution: Safe to ignore for now; does not affect workflow.
- **Preprocessing script unpacking error:**
  - Error: `ValueError: not enough values to unpack (expected 3, got 2)`
  - Solution: Fixed train/val split logic in script.

## How to Run
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Preprocess data:
   ```
   python src/preprocess.py
   ```
3. Track processed data with DVC:
   ```
   dvc add data/processed
   git add data/processed.dvc .gitignore
   git commit -m "Track processed data with DVC"
   ```
4. Train model:
   ```
   python src/train.py
   ```
5. Evaluate model:
   ```
   python src/evaluate.py
   ```
6. Run MLflow UI:
   ```
   mlflow ui
   ```
7. Run inference API:
   ```
   uvicorn src.app:app --reload
   ```

## Next Steps
- Add unit tests for preprocessing and inference
- Create Dockerfile for API containerization
- Set up CI/CD pipeline
- Add deployment manifests (Kubernetes/Docker Compose)
- Implement monitoring and logging

---
For any issues, check the troubleshooting section above or reach out for help.
