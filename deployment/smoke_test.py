import requests
import time

API_URL = "http://localhost:30080"

# Health check
try:
    resp = requests.get(f"{API_URL}/health", timeout=5)
    assert resp.status_code == 200
    print("Health endpoint OK:", resp.json())
except Exception as e:
    print("Health endpoint FAILED:", str(e))
    exit(1)

# Prediction check (using a sample image)
try:
    with open("deployment/sample.jpg", "rb") as f:
        files = {"file": f}
        resp = requests.post(f"{API_URL}/predict", files=files, timeout=10)
        assert resp.status_code == 200
        print("Predict endpoint OK:", resp.json())
except Exception as e:
    print("Predict endpoint FAILED:", str(e))
    exit(1)

print("Smoke tests PASSED.")
