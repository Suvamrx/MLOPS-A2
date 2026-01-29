# End-to-end MLOps pipeline for binary image classification (Cats vs Dogs) using PyTorch, FastAPI, DVC, Docker, and GitHub Actions. Includes data versioning, preprocessing, model training, experiment tracking, API packaging, containerization, automated testing, and CI/CD deployment. Suitable for pet adoption platforms and MLOps coursework.
# Cats vs Dogs MLOps Pipeline

## Author : **Suvam Pattnaik**
## Project Overview
End-to-end MLOps pipeline for binary image classification (Cats vs Dogs) for a pet adoption platform. Includes data versioning, model training, experiment tracking, packaging, containerization, and CI/CD setup.

## Folder Structure
- `.dvc/` — DVC internal files and cache
- `.github/workflows/` — CI/CD pipeline workflows (ci.yml, cd.yml)
- `.vscode/` — VS Code settings
- `data/` — Raw and processed datasets (DVC tracked)
  - `raw/` — Original dataset
    - `training_set/cats/`, `training_set/dogs/`, `test_set/cats/`, `test_set/dogs/`
  - `processed/` — Preprocessed images
    - `train/cat/`, `train/dog/`, `val/cat/`, `val/dog/`, `test/cat/`, `test/dog/`
- `deployment/` — Kubernetes manifests, smoke test script, sample image
  - `deployment.yaml`, `service.yaml`, `smoke_test.py`, `batch_performance.py`, `batch_results.json`, `batch_test_images/`
- `models/` — Saved model files (e.g., `cnn.pt`)
- `src/` — Python source code
  - `app.py`, `evaluate.py`, `preprocess.py`, `train.py`
- `tests/` — Unit tests for preprocessing and model
  - `test_preprocess.py`, `test_model.py`
- `Dockerfile` — Containerization for inference API
- `README.md` — Project documentation
- `requirements.txt` — Python dependencies
- `.gitignore` — Git ignore rules
- `mlruns/` — MLflow experiment tracking

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
- Configured workflow to push Docker image to GitHub Container Registry (GHCR)
- Fixed image tag case and permissions issues for GHCR
- Pushed code to GitHub to trigger CI pipeline

### 8. Kubernetes Deployment
- Created `deployment/deployment.yaml` and `deployment/service.yaml` for Kubernetes
- Applied manifests with `kubectl apply -f ...` to deploy the API on Docker Desktop Kubernetes
- Verified pod and service status with `kubectl get pods` and `kubectl get svc`
- Troubleshooted pod startup and image pull issues (waited for image pull, checked pod events)
- Confirmed API is accessible at `http://localhost:30080/health`

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

## Continuous Deployment (CD) Notes

This project includes a GitHub Actions workflow (`.github/workflows/cd.yml`) for automated deployment to Kubernetes and post-deployment smoke tests. For cloud clusters (GKE, EKS, AKS), this workflow can deploy directly from GitHub Actions using a kubeconfig secret.

## Post-Deployment Model Performance Tracking

To evaluate model performance after deployment, run a batch of test images through the deployed API and compare predictions to true labels:

1. Place a set of labeled test images in `deployment/batch_test_images/`.
2. Edit `deployment/batch_performance.py` to list the filenames and true labels in the `image_labels` list.
3. Run the script:
  ```
  python deployment/batch_performance.py
  ```
4. The script will log predictions, save results to `deployment/batch_results.json`, and print batch accuracy.

This satisfies the post-deployment model performance tracking requirement of the assignment.

**Local Cluster Limitation:**
For local clusters (Docker Desktop, minikube), the cluster is not accessible from GitHub Actions runners. Therefore, the deployment step must be run manually:

```
kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/service.yaml
```

After deployment, run the smoke test script locally to verify health and prediction endpoints:

```
python deployment/smoke_test.py
```

This approach demonstrates the intended CD automation and provides a manual workaround for local environments, satisfying assignment requirements.

## How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Preprocess data
```
python src/preprocess.py
```

### 3. Track processed data with DVC
```
dvc add data/processed
git add data/processed.dvc .gitignore
git commit -m "Track processed data with DVC"
```

### 4. Train the model
```
python src/train.py
```

### 5. Evaluate the model
```
python src/evaluate.py
```

### 6. Run MLflow UI (optional)
```
mlflow ui
```

### 7. Run inference API locally
```
uvicorn src.app:app --reload
```

### 8. Build and run Docker container
```
docker build -t cats-vs-dogs-api .
docker run -p 8000:8000 cats-vs-dogs-api
```

### 9. Deploy to Kubernetes (local cluster)
```
kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/service.yaml
```

### 10. Run smoke tests
```
python deployment/smoke_test.py
```

