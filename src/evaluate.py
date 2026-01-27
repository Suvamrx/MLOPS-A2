"""
Evaluate trained CNN model on test set and log results to MLflow.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch

DATA_DIR = 'data/processed'
MODEL_PATH = 'models/cnn.pt'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data transform for test set
test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def get_test_loader():
    test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    return test_loader

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loader = get_test_loader()
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    test_acc = evaluate(model, test_loader, device)
    print(f'Test Accuracy: {test_acc:.3f}')
    mlflow.set_experiment('cats-vs-dogs-cnn')
    with mlflow.start_run(run_name='test-eval'):
        mlflow.log_metric('test_acc', test_acc)

if __name__ == '__main__':
    main()
