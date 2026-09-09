"""Honest evaluation of the Happy/Sad classifier: 5-fold stratified cross-validation.

Why this exists
---------------
The notebooks evaluate on a single 480/120 split. With 120 validation images a
single image is worth 0.83 points, so differences under ~8 points between
experiments are not measurable, and "best epoch on the validation set" is itself
a way of fitting the thing you are measuring against.

This script instead predicts every one of the 600 images exactly once, each time
by a model that never saw it during training, and reports mean +/- std across
the 5 folds.

Results on this dataset (PyTorch 2.14, CPU, seed 42):

    DeepCNN from scratch  @64                 60.67% +/- 3.70   (127s)
    ResNet18 fine-tuned   @64                 79.17% +/- 2.04   ( 80s)
    ResNet18 fine-tuned   @224 + ImageNet     87.33% +/- 0.62   (509s)

Two things that comparison isolates:

1. Pretraining is worth ~18 points over training from scratch on 600 images.
2. Feeding the pretrained backbone 224px instead of 64px is worth ~8 more,
   even though the source images are 48x48 and the upscaling adds no
   information. An ImageNet backbone's strides and receptive fields expect
   that scale; at 64px its deep layers see almost nothing.

The original notebook's Experiment 7 froze the backbone and scored 65.8%. The
same idea fully fine-tuned at the same resolution reaches 79.2% -- freezing,
not resolution, was that experiment's mistake.

Usage
-----
    python experiments/cross_validation.py              # all three configs
    python experiments/cross_validation.py ResNet18     # substring filter
"""

import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights

SEED = 42
DATA = "data"
N_FOLDS = 5
IMAGENET_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HALF_STATS = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # what the notebooks used

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stratified_folds(labels, n_folds=N_FOLDS, seed=SEED):
    """Assign each index a fold, keeping the class balance even across folds."""
    rng = np.random.RandomState(seed)
    folds = np.zeros(len(labels), dtype=int)
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        for position, i in enumerate(idx):
            folds[i] = position % n_folds
    return folds


def transforms_for(size, stats, train):
    if not train:
        return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(*stats)])
    # Horizontal flip is the high-value augmentation here: faces are near-symmetric.
    # Hue/saturation jitter would be wasted -- the source images are grayscale.
    return T.Compose([
        T.Resize((size, size)),
        T.RandomHorizontalFlip(),
        T.RandomAffine(10, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(*stats),
    ])


class DeepCNN(nn.Module):
    """The notebook's best from-scratch architecture (Experiment 5)."""

    def __init__(self):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32)
        self.conv2, self.bn2 = nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64)
        self.conv3, self.bn3 = nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128)
        self.relu, self.pool = nn.ReLU(), nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1, self.fc2 = nn.Linear(128 * 8 * 8, 256), nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x.view(x.size(0), -1))
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


def build(kind):
    if kind == "scratch":
        return DeepCNN()
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def run_fold(kind, size, stats, epochs, lr, folds, labels, k):
    train_idx = np.where(folds != k)[0]
    val_idx = np.where(folds == k)[0]
    assert not set(train_idx) & set(val_idx), "train/val overlap"

    train_loader = DataLoader(
        Subset(ImageFolder(DATA, transform=transforms_for(size, stats, True)), train_idx),
        batch_size=32, shuffle=True)
    val_loader = DataLoader(
        Subset(ImageFolder(DATA, transform=transforms_for(size, stats, False)), val_idx),
        batch_size=32)

    torch.manual_seed(SEED + k)
    model = build(kind).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * len(train_loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
            scheduler.step()

    # Test-time augmentation: average the prediction with its mirror image.
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            probs = torch.softmax(model(x), 1) + torch.softmax(model(torch.flip(x, [3])), 1)
            correct += (probs.argmax(1) == y).sum().item()
            total += y.size(0)
    return 100 * correct / total


CONFIGS = [
    # label,                                        kind,      size, stats,          epochs, lr
    ("DeepCNN from scratch  @64",                   "scratch",   64, HALF_STATS,      20, 1e-3),
    ("ResNet18 fine-tuned   @64",                   "r18",       64, HALF_STATS,       8, 3e-4),
    ("ResNet18 fine-tuned   @224 + ImageNet stats", "r18",      224, IMAGENET_STATS,   8, 3e-4),
]


def main():
    dataset = ImageFolder(DATA)
    labels = np.array([y for _, y in dataset.samples])
    folds = stratified_folds(labels)

    print(f"device: {device}")
    print(f"{len(labels)} images | classes {dataset.classes} | "
          f"balance {np.bincount(labels).tolist()} | {N_FOLDS}-fold stratified\n")

    wanted = sys.argv[1] if len(sys.argv) > 1 else None
    for label, kind, size, stats, epochs, lr in CONFIGS:
        if wanted and wanted.lower() not in label.lower():
            continue
        print(label)
        start = time.time()
        accs = [run_fold(kind, size, stats, epochs, lr, folds, labels, k) for k in range(N_FOLDS)]
        for k, acc in enumerate(accs):
            print(f"    fold {k}: {acc:.2f}%")
        print(f"  >> {np.mean(accs):.2f}% +/- {np.std(accs):.2f}  ({time.time() - start:.0f}s)\n")


if __name__ == "__main__":
    main()
