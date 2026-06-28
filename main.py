import os
import csv
import time
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image
from sklearn.metrics import confusion_matrix

from baseline_model import BaselineCNN
from resnet_model import ResNet18CIFAR
from transformer_model import VisionTransformer

# --------------------
# CONFIG
# --------------------

MODEL = "vit"   # "baseline" or "resnet" or "vit"
SEED = 1            # 0, 1, 42.

BATCH_SIZE = 128
NUM_EPOCHS = 80

RUN_ROOT = "runs/cnn_vs_resnet"

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


# --------------------
# UTILS
# --------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def save_failures(model, dataloader, run_dir, max_to_save=100, device="cpu"):
    fail_dir = os.path.join(run_dir, "failures")
    os.makedirs(fail_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "failures.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["true_id", "true_name", "pred_id", "pred_name", "confidence", "filename"]
        )

    model.eval()
    saved = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        probs = F.softmax(logits, dim=1)

        preds = probs.argmax(dim=1)
        confs = probs.max(dim=1).values

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

            filename = (
                f"fail_{saved:04d}_true_{true_name}_pred_{pred_name}_conf{conf:.2f}.png"
            )
            path = os.path.join(fail_dir, filename)

            save_image(mis_images[i].detach().cpu().clamp(0, 1), path)

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(
                    f
                )
                writer.writerow(
                    [true_id, true_name, pred_id, pred_name, conf, filename]
                )
            saved += 1


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device="cpu"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


# CutMix helpers kept for reference, but disabled by default
def rand_bbox(W, H, lam):
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return x1, y1, x2, y2


def cutmix(images, labels, alpha=1.0):
    if alpha <= 0:
        return images, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    y_a = labels
    y_b = labels[index]

    _, _, H, W = images.shape
    x1, y1, x2, y2 = rand_bbox(W, H, lam)

    images = images.clone()
    images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
    return images, y_a, y_b, lam


# --------------------
# MAIN
# --------------------

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- dataset & transforms ----
    base_dataset = CIFAR10(root="data", train=True, download=True, transform=None)

    val_fraction = 0.1
    val_size = int(len(base_dataset) * val_fraction)
    train_size = len(base_dataset) - val_size

    train_subset, val_subset = random_split(
        range(len(base_dataset)),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    normalize = transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    test_dataset = CIFAR10(
        root="data", train=False, download=True, transform=test_transform
    )
    base_train = CIFAR10(
        root="data", train=True, download=False, transform=train_transform
    )
    base_val = CIFAR10(
        root="data", train=True, download=False, transform=test_transform
    )

    train_indices = train_subset.indices
    val_indices = val_subset.indices

    train_split = Subset(base_train, train_indices)
    val_split = Subset(base_val, val_indices)

    train_loader = DataLoader(
        train_split, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_split, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # ---- model choice ----
    if MODEL == "baseline":
        model = BaselineCNN().to(device)
        exp_name = "adamw_cosine_ls0.1_bn_fc128_do0.3_conv4_128"
    elif MODEL == "resnet":
        model = ResNet18CIFAR().to(device)
        exp_name = "adamw_cosine_ls0.1_resnet18"
    elif MODEL == "vit":
        model = VisionTransformer(
            img_size=32,
            patch_size=4,
            in_chans=3,
            num_classes=10,
            embed_dim=128,
            depth=6,
            num_heads=4,
            mlp_ratio=4.0,
            dropout=0.1,
            attn_dropout=0.1,
        ).to(device)
        exp_name = "adamw_cosine_ls0.1_vit_tiny"
    else:
        raise ValueError(f"Unknown MODEL: {MODEL}")


    # ---- loss, optimizer, scheduler ----
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=5e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
    )

    patience = 15
    min_delta = 0.0

    best_val_loss = float("inf")
    best_state = None
    best_val_acc = 0.0
    epochs_no_improve = 0

    # ---- run directory + logging ----
    run_dir = os.path.join(RUN_ROOT, exp_name, f"seed_{SEED}")
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time"]
        )

    # CutMix turned OFF for final baseline
    use_cutmix = False
    cutmix_prob = 0.25
    cutmix_alpha = 0.5

    def cutmix_criterion(criterion, preds, y_a, y_b, lam):
        return lam * criterion(preds, y_a) + (1.0 - lam) * criterion(preds, y_b)

    # --------------------
    # TRAIN LOOP
    # --------------------
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        model.train()
        train_loss_sum = 0.0
        train_correct = 0.0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            did_cutmix = False

            if use_cutmix and np.random.rand() < cutmix_prob:
                did_cutmix = True
                images_mixed, y_a, y_b, lam = cutmix(
                    images, labels, alpha=cutmix_alpha
                )
                logits = model(images_mixed)
                loss = cutmix_criterion(criterion, logits, y_a, y_b, lam)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            if did_cutmix:
                train_correct += (
                    lam * (preds == y_a).float()
                    + (1.0 - lam) * (preds == y_b).float()
                ).sum().item()
            else:
                train_correct += (preds == labels).sum().item()

            train_total += labels.size(0)
            train_loss_sum += loss.item() * labels.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        val_loss, val_acc = evaluate(model, val_loader, criterion, device=device)

        epoch_time = time.time() - start_time

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc, epoch_time])

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.4f} | "
            f"time {epoch_time:.2f}s"
        )

        scheduler.step()

        # early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best val loss: {best_val_loss:.4f}, acc: {best_val_acc:.4f}"
            )
            break

    # --------------------
    # EVAL: FAILURES, CM, TEST
    # --------------------
    if best_state is not None:
        model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt")))
        save_failures(model, val_loader, run_dir, device=device)

    # Confusion matrix on val
    all_true = []
    all_pred = []

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)

            all_true.append(labels)
            all_pred.append(preds)

    y_true = torch.cat(all_true).cpu()
    y_pred = torch.cat(all_pred).cpu()
    cm = confusion_matrix(y_true.numpy(), y_pred.numpy())
    acc_from_cm = np.trace(cm) / np.sum(cm)

    np.savetxt(
        os.path.join(run_dir, "confusion_matrix.csv"),
        cm,
        fmt="%d",
        delimiter=",",
    )

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (Validation)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    # Test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device=device)

    print(f"TEST loss: {test_loss:.4f}")
    print(f"TEST acc : {test_acc:.4f}")

    test_results_path = os.path.join(run_dir, "test_results.csv")
    with open(test_results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_loss", "test_acc", "val_acc_from_cm"])
        writer.writerow([test_loss, test_acc, acc_from_cm])

    # Global summary
    summary_path = os.path.join(RUN_ROOT, "summary.csv")
    file_exists = os.path.isfile(summary_path)

    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "experiment",
                    "model",
                    "seed",
                    "normalize",
                    "augmentation",
                    "optimizer",
                    "lr",
                    "weight_decay",
                    "epochs_trained",
                    "best_val_loss",
                    "best_val_acc",
                    "test_acc",
                ]
            )

        writer.writerow(
            [
                exp_name,
                MODEL,
                SEED,
                True,
                True,
                "AdamW",
                1e-3,
                5e-4,
                epoch + 1,
                best_val_loss,
                best_val_acc,
                test_acc,
            ]
        )


if __name__ == "__main__":
    main()
