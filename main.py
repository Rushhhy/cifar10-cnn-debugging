import os, csv

import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import transforms
from baseline_model import BaselineCNN
import torch.nn as nn

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import copy

from torchvision.utils import save_image
import torch.nn.functional as F

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

@torch.no_grad()
def save_failures(cnn_model, dataloader, run_dir, max_to_save=100):
    fail_dir = os.path.join(run_dir, "failures")
    os.makedirs(fail_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "failures.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_id","true_name","pred_id","pred_name","confidence","filename"])

    cnn_model.eval()
    saved = 0
    for batch_idx, (images, labels) in enumerate(dataloader):
        logits = model(images)
        probs = F.softmax(logits, dim=1) # turns logits to probs

        preds = probs.argmax(dim=1)
        confs = probs.max(dim=1).values # confidence of predicted class

        mis_mask = preds != labels
        if not mis_mask.any():
            continue

        mis_images = images[mis_mask]
        mis_labels = labels[mis_mask]
        mis_preds = preds[mis_mask]
        mis_confs = confs[mis_mask]

        for i in range(mis_images.size(0)):
            if saved >= max_to_save:
                return
            true_id = int(mis_labels[i].item())
            pred_id = int(mis_preds[i].item())
            conf = float(mis_confs[i].item())

            true_name = CLASS_NAMES[true_id]
            pred_name = CLASS_NAMES[pred_id]

            filename =  filename = f"fail_{saved:04d}_true_{true_name}_pred_{pred_name}_conf{conf:.2f}.png"
            path = os.path.join(fail_dir, filename)

            #save the image tensor to disk
            save_image(mis_images[i].clamp(0, 1), path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([true_id, true_name, pred_id, pred_name, conf, filename])

            saved += 1
@torch.no_grad()
def evaluate(cnn_model, dataloader, loss_fn):
    cnn_model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        logits = cnn_model(images)
        loss = loss_fn(logits, labels)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


base_dataset = CIFAR10(root="data", train=True, download=True, transform=None)

val_fraction = 0.1
val_size = int(len(base_dataset) * val_fraction)
train_size = len(base_dataset) - val_size

train_subset, val_subset = random_split(
    range(len(base_dataset)),
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

normalize = transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
train_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

test_dataset = CIFAR10(root="data", train=False, download=True, transform=test_transform)
base_train = CIFAR10(root="data", train=True, download=False, transform=train_transform)
base_val = CIFAR10(root="data", train=True, download=False, transform=test_transform)

train_indices = train_subset.indices
val_indices = val_subset.indices

train_split = Subset(base_train, train_indices)
val_split = Subset(base_val, val_indices)

train_loader = DataLoader(
    train_split,
    batch_size=128,
    shuffle=True
)
val_loader = DataLoader(
    val_split,
    batch_size=128,
    shuffle=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)

model = BaselineCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 40

patience = 6
min_delta = 1e-4

best_val_loss = float("inf")
best_state = None
epochs_no_improve = 0

run_dir = "runs/baseline_norm"
os.makedirs(run_dir, exist_ok=True)

log_path = os.path.join(run_dir, "train_log.csv")
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

for epoch in range(num_epochs):
    model.train()
    train_loss_sum = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)
        train_loss_sum += loss.item() * labels.size(0)

    train_loss = train_loss_sum / train_total
    train_acc = train_correct / train_total

    val_loss, val_acc = evaluate(model, val_loader, criterion)

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc])

    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"training set loss {train_loss:.4f}, accuracy {train_acc:.4f} | "
        f"validation set loss {val_loss:.4f}, accuracy {val_acc:.4f}"
    )

    # early stopping
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered at epoch {epoch + 1}. Best val loss: {best_val_loss:.4f}")
        break

# restore best weights
if best_state is not None:
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt")))
    save_failures(model, val_loader, run_dir)

# Confusion Matrix
all_true = []
all_pred = []

model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        logits = model(images)
        preds = logits.argmax(dim=1)

        all_true.append(labels)
        all_pred.append(preds)

y_true = torch.cat(all_true)
y_pred = torch.cat(all_pred)

cm = confusion_matrix(y_true.numpy(), y_pred.numpy())
acc_from_cm = np.trace(cm) / np.sum(cm)

np.savetxt(
    os.path.join(run_dir, "confusion_matrix.csv"),
    cm,
    fmt="%d",
    delimiter=","
)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix(Validation)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "confusion_matrix.png"), dpi=200)
plt.close()

test_loss, test_acc = evaluate(model, test_loader, criterion)

print(f"TEST loss: {test_loss:.4f}")
print(f"TEST acc : {test_acc:.4f}")

test_results_path = os.path.join(run_dir, "test_results.csv")
with open(test_results_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["test_loss", "test_acc"])
    writer.writerow([test_loss, test_acc])