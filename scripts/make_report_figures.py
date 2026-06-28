from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"

FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)


def clean_summary(df):
    df = df.copy()

    # Normalize true/false formatting
    for col in ["normalize", "augmentation"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()

    # Convert numeric columns
    numeric_cols = [
        "seed",
        "lr",
        "weight_decay",
        "epochs_trained",
        "best_val_loss",
        "best_val_acc",
        "test_acc",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace("N/A", np.nan), errors="coerce")

    # Remove duplicate experiment+seed rows.
    if "model" in df.columns:
        df = df.drop_duplicates(subset=["experiment", "model", "seed"], keep="last")
    else:
        df = df.drop_duplicates(subset=["experiment", "seed"], keep="last")

    return df


def save_table(df, filename):
    path = TABLES_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved table: {path}")


def format_acc(x):
    return x * 100


def save_bar_chart(df, x_col, y_col, title, ylabel, filename, rotation=35):
    plt.figure(figsize=(11, 6))
    plt.bar(df[x_col], df[y_col])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right")
    plt.ylim(max(0, df[y_col].min() - 5), min(100, df[y_col].max() + 5))
    plt.tight_layout()

    path = FIGURES_DIR / filename
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved figure: {path}")


def save_error_bar_chart(df, x_col, mean_col, std_col, title, ylabel, filename):
    plt.figure(figsize=(11, 6))
    plt.bar(df[x_col], df[mean_col], yerr=df[std_col], capsize=5)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.ylim(max(0, df[mean_col].min() - 5), min(100, df[mean_col].max() + 5))
    plt.tight_layout()

    path = FIGURES_DIR / filename
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved figure: {path}")


def plot_train_log(train_log_path, output_name):
    df = pd.read_csv(train_log_path)

    # Try to detect common column names.
    epoch_col = "epoch" if "epoch" in df.columns else df.columns[0]

    possible_acc_cols = [
        ("train_acc", "val_acc"),
        ("train_accuracy", "val_accuracy"),
        ("accuracy", "val_accuracy"),
    ]

    possible_loss_cols = [
        ("train_loss", "val_loss"),
        ("loss", "val_loss"),
    ]

    acc_pair = None
    loss_pair = None

    for train_col, val_col in possible_acc_cols:
        if train_col in df.columns and val_col in df.columns:
            acc_pair = (train_col, val_col)
            break

    for train_col, val_col in possible_loss_cols:
        if train_col in df.columns and val_col in df.columns:
            loss_pair = (train_col, val_col)
            break

    if acc_pair:
        plt.figure(figsize=(9, 5))
        plt.plot(df[epoch_col], df[acc_pair[0]], label="Train accuracy")
        plt.plot(df[epoch_col], df[acc_pair[1]], label="Validation accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()

        path = FIGURES_DIR / f"{output_name}_accuracy_curve.png"
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"Saved figure: {path}")

    if loss_pair:
        plt.figure(figsize=(9, 5))
        plt.plot(df[epoch_col], df[loss_pair[0]], label="Train loss")
        plt.plot(df[epoch_col], df[loss_pair[1]], label="Validation loss")
        plt.title("Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()

        path = FIGURES_DIR / f"{output_name}_loss_curve.png"
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"Saved figure: {path}")


def plot_confusion_matrix(csv_path, output_name):
    cm = pd.read_csv(csv_path, index_col=0)

    plt.figure(figsize=(9, 8))
    plt.imshow(cm.values)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(len(cm.columns)), cm.columns, rotation=45, ha="right")
    plt.yticks(range(len(cm.index)), cm.index)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm.values[i, j]
            plt.text(j, i, str(value), ha="center", va="center", fontsize=8)

    plt.colorbar()
    plt.tight_layout()

    path = FIGURES_DIR / f"{output_name}_confusion_matrix.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved figure: {path}")


def main():
    # ---------------------------
    # 1. Main CNN optimization summary
    # ---------------------------
    main_summary_path = RUNS_DIR / "summary.csv"

    if not main_summary_path.exists():
        raise FileNotFoundError(f"Could not find {main_summary_path}")

    summary = clean_summary(pd.read_csv(main_summary_path))
    save_table(summary, "clean_summary.csv")

    # Single-seed optimization path, mostly seed 42.
    optimization_order = [
        "baseline_no_norm",
        "baseline_norm",
        "baseline_norm_aug",
        "batchnorm",
        "bn_fc128_dropout0.3",
        "bn_fc128_do0.3_conv4_128",
        "adamw_cosine_bn_fc128_do0.3_conv4_128",
        "adamw_cosine_ls0.1_bn_fc128_do0.3_conv4_128",
        "adamw_cosine_cutmix_ls0.0_bn_fc128_do0.3_conv4_128",
    ]

    readable_names = {
        "baseline_no_norm": "Baseline",
        "baseline_norm": "Normalization",
        "baseline_norm_aug": "Augmentation",
        "adam_lr3e-4": "Lower LR Adam",
        "sgd_lr0.1": "SGD",
        "batchnorm": "BatchNorm",
        "bn_fc128_dropout0.3": "BatchNorm + Dropout",
        "bn_fc128_do0.3_conv4_128": "Deeper CNN",
        "adamw_cosine_bn_fc128_do0.3_conv4_128": "AdamW + Cosine",
        "adamw_cosine_ls0.1_bn_fc128_do0.3_conv4_128": "Label Smoothing",
        "adamw_cosine_ls0.1_bn_fc128_do0.3_conv4_128_erasing": "Random Erasing",
        "adamw_cosine_cutmix_ls0.0_bn_fc128_do0.3_conv4_128": "CutMix",
    }

    path_rows = []

    for exp in optimization_order:
        rows = summary[summary["experiment"] == exp]

        if rows.empty:
            continue

        seed_42 = rows[rows["seed"] == 42]
        row = seed_42.iloc[-1] if not seed_42.empty else rows.sort_values("test_acc").iloc[-1]

        path_rows.append({
            "experiment": readable_names.get(exp, exp),
            "test_acc_percent": format_acc(row["test_acc"]),
        })

    optimization_df = pd.DataFrame(path_rows)
    save_table(optimization_df, "optimization_path.csv")

    save_bar_chart(
        optimization_df,
        "experiment",
        "test_acc_percent",
        "CNN Optimization Path",
        "Test accuracy (%)",
        "01_cnn_optimization_path.png",
    )

    # ---------------------------
    # 2. Multi-seed CNN comparison
    # ---------------------------
    final_cnn_experiments = [
        "bn_fc128_do0.3_conv4_128",
        "adamw_cosine_bn_fc128_do0.3_conv4_128",
        "adamw_cosine_ls0.1_bn_fc128_do0.3_conv4_128",
        "adamw_cosine_ls0.1_bn_fc128_do0.3_conv4_128_erasing",
        "adamw_cosine_cutmix_ls0.0_bn_fc128_do0.3_conv4_128",
    ]

    final_cnn = summary[summary["experiment"].isin(final_cnn_experiments)].copy()

    grouped = (
        final_cnn
        .groupby("experiment")
        .agg(
            mean_test_acc=("test_acc", "mean"),
            std_test_acc=("test_acc", "std"),
            best_test_acc=("test_acc", "max"),
            runs=("test_acc", "count"),
        )
        .reset_index()
    )

    grouped["mean_test_acc_percent"] = grouped["mean_test_acc"] * 100
    grouped["std_test_acc_percent"] = grouped["std_test_acc"] * 100
    grouped["best_test_acc_percent"] = grouped["best_test_acc"] * 100
    grouped["experiment"] = grouped["experiment"].map(readable_names).fillna(grouped["experiment"])

    save_table(grouped, "final_cnn_variant_summary.csv")

    save_error_bar_chart(
        grouped,
        "experiment",
        "mean_test_acc_percent",
        "std_test_acc_percent",
        "Final CNN Variants Across Seeds",
        "Mean test accuracy (%)",
        "02_final_cnn_variants.png",
    )

    # ---------------------------
    # 3. CNN vs ResNet vs ViT summary
    # ---------------------------
    model_summary_path = RUNS_DIR / "cnn_vs_resnet" / "summary.csv"

    if model_summary_path.exists():
        model_summary = clean_summary(pd.read_csv(model_summary_path))
        save_table(model_summary, "clean_cnn_vs_resnet_summary.csv")

        model_grouped = (
            model_summary
            .groupby("model")
            .agg(
                mean_val_acc=("best_val_acc", "mean"),
                mean_test_acc=("test_acc", "mean"),
                std_test_acc=("test_acc", "std"),
                best_test_acc=("test_acc", "max"),
                runs=("test_acc", "count"),
            )
            .reset_index()
        )

        model_grouped["mean_val_acc_percent"] = model_grouped["mean_val_acc"] * 100
        model_grouped["mean_test_acc_percent"] = model_grouped["mean_test_acc"] * 100
        model_grouped["std_test_acc_percent"] = model_grouped["std_test_acc"] * 100
        model_grouped["best_test_acc_percent"] = model_grouped["best_test_acc"] * 100

        model_names = {
            "baseline": "Custom CNN",
            "resnet": "ResNet18",
            "vit": "ViT Tiny",
        }

        model_grouped["model"] = model_grouped["model"].map(model_names).fillna(model_grouped["model"])

        save_table(model_grouped, "model_comparison_summary.csv")

        save_error_bar_chart(
            model_grouped,
            "model",
            "mean_test_acc_percent",
            "std_test_acc_percent",
            "Custom CNN vs ResNet18 vs ViT Tiny",
            "Mean test accuracy (%)",
            "03_model_comparison.png",
        )
    else:
        print(f"Skipped model comparison. Missing file: {model_summary_path}")

    # ---------------------------
    # 4. Best custom CNN training curves and confusion matrix
    # ---------------------------
    best_custom_cnn_dir = (
            RUNS_DIR
            / "cnn_vs_resnet"
            / "adamw_cosine_ls0.1_bn_fc128_do0.3_conv4_128"
            / "seed_0"
    )

    if (best_custom_cnn_dir / "train_log.csv").exists():
        plot_train_log(
            best_custom_cnn_dir / "train_log.csv",
            "04_best_custom_cnn_seed0"
        )

    if (best_custom_cnn_dir / "confusion_matrix.csv").exists():
        plot_confusion_matrix(
            best_custom_cnn_dir / "confusion_matrix.csv",
            "05_best_custom_cnn_seed0"
        )

    if (best_custom_cnn_dir / "confusion_matrix.png").exists():
        copied_path = FIGURES_DIR / "05_best_custom_cnn_seed0_confusion_matrix_original.png"
        shutil.copy(best_custom_cnn_dir / "confusion_matrix.png", copied_path)
        print(f"Copied figure: {copied_path}")


if __name__ == "__main__":
    main()