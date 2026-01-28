import requests
import os
import json

API_URL = "http://localhost:30080/predict"
TEST_IMAGES_DIR = "deployment/batch_test_images"
RESULTS_FILE = "deployment/batch_results.json"

# Prepare a list of (filename, true_label) pairs
# Example: [('cat1.jpg', 'cat'), ('dog1.jpg', 'dog'), ...]
image_labels = [
    ('cat.4005.jpg', 'cat'),
    ('cat.4008.jpg', 'cat'),
    ('dog.4002.jpg', 'dog'),
    ('dog.4007.jpg', 'dog'),
]

results = []
for fname, true_label in image_labels:
    img_path = os.path.join(TEST_IMAGES_DIR, fname)
    with open(img_path, "rb") as f:
        files = {"file": f}
        resp = requests.post(API_URL, files=files, timeout=10)
        pred = resp.json().get('label', 'error')
        results.append({
            "filename": fname,
            "true_label": true_label,
            "predicted_label": pred,
            "response": resp.json()
        })
    print(f"{fname}: true={true_label}, predicted={pred}")

# Save results to file
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

# Print summary accuracy
correct = sum(1 for r in results if r['true_label'] == r['predicted_label'])
print(f"Batch accuracy: {correct}/{len(results)} = {correct/len(results):.2f}")
