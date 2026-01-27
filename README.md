# End-to-end MLOps pipeline for binary image classification (Cats vs Dogs) using PyTorch, FastAPI, DVC, Docker, and GitHub Actions. Includes data versioning, preprocessing, model training, experiment tracking, API packaging, containerization, automated testing, and CI/CD deployment. Suitable for pet adoption platforms and MLOps coursework.
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
- Initialized local git repo and pushed to GitHub

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
- Added FastAPI, Uvicorn, and python-multipart to `requirements.txt`
- Created `src/app.py` for REST API with `/health` and `/predict` endpoints
- Tested API locally and in Docker

### 6. Docker Packaging
- Created `Dockerfile` for containerizing the FastAPI inference service
- Built and ran Docker image locally

### 7. Automated Testing & CI
- Added unit tests for preprocessing and model inference in `tests/`
- Set up GitHub Actions workflow for CI: installs dependencies, runs tests, builds Docker image
- Pushed code to GitHub to trigger CI pipeline

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

- **Pytest cannot find src modules in CI:**
   - Error: `ModuleNotFoundError: No module named 'src'`
   - Solution: Added a step to set PYTHONPATH to include src in the GitHub Actions workflow.
- **Docker image tag must be lowercase for GHCR:**
   - Error: `invalid tag ... repository name must be lowercase`
   - Solution: Updated workflow to convert the image tag to all lowercase before building and pushing.

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

1. Deployment Automation
   - Add deployment manifests: Kubernetes YAML (Deployment, Service) or docker-compose.yml for local/VM deployment.
   - Set up CD pipeline (GitHub Actions, ArgoCD, or Jenkins) to automate deployment on main branch changes.
   - Implement post-deploy smoke tests (health and prediction endpoint checks).

2. Monitoring & Logging
   - Enable request/response logging in the inference API (excluding sensitive data).
   - Track basic metrics: request count, latency (via logs, Prometheus, or in-app counters).
   - Collect and log model performance metrics post-deployment.

3. Final Submission
   - Prepare a zip file with all source code, configuration files, trained model artifacts, and documentation.
   - Record a short screen demo (<5 min) showing the MLOps workflow from code change to deployed prediction.

---
For any issues, check the troubleshooting section above or reach out for help.
