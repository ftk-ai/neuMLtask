"""Kaggle: run one selected preprocessing variant with stratified 5-fold CV.

Run in a notebook: %run /kaggle/working/compare_preprocessing_cv.py
Requires torch, torchvision, numpy, opencv-python, Pillow, scikit-learn.
ImageNet weights are downloaded by torchvision on first use (enable Internet),
or supplied through IMAGENET_WEIGHTS_PATH below.
"""

from pathlib import Path

# ==================== Paths: edit these first ====================
TRAIN_DIR = Path(
    "/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2/train"
)
OUTPUT_DIR = Path("/kaggle/working/preprocessing_comparison")
# Optional local torchvision ResNet18 ImageNet state_dict (.pth).
# Leave None to use models.ResNet18_Weights.IMAGENET1K_V1.
IMAGENET_WEIGHTS_PATH = None

# ==================== Experiment configuration ====================
# Change ONLY this setting to choose the dataset for this run:
# "original": original background; 
# "black": black background;
# "imagenet": ImageNet mean-color background.
DATA_VERSION = "black"

# 冻结策略：每次只运行所选策略；三种策略均更新训练阶段的 BN 运行统计量。
# 1：只训练最后的分类层（卷积层、BN 的 gamma/beta 参数冻结）。
# 2：训练分类层 + 所有 BN 的 gamma/beta（与之前代码一致）。
# 3：全部参数参与训练（仍使用 ImageNet 预训练初始化）。
FREEZE_STRATEGY = 3

# 数据增强开关：True 使用原来的全部增强；False 不使用随机增强。
# 两种情况都保留 ToTensor 和 ImageNet 标准化；验证集始终不做随机增强。
USE_DATA_AUGMENTATION = False

# Internal settings: each run uses exactly one version and its own output folder.
if DATA_VERSION not in {"original", "black", "imagenet"}:
    raise ValueError('DATA_VERSION must be "original", "black", or "imagenet"')
FREEZE_STRATEGY_NAMES = {1: "fc_only", 2: "fc_bn", 3: "full_finetune"}
if type(FREEZE_STRATEGY) is not int or FREEZE_STRATEGY not in FREEZE_STRATEGY_NAMES:
    raise ValueError("FREEZE_STRATEGY must be 1, 2, or 3")
EXPERIMENTS = (DATA_VERSION,)
OUTPUT_DIR = OUTPUT_DIR / DATA_VERSION / f"strategy_{FREEZE_STRATEGY}_{FREEZE_STRATEGY_NAMES[FREEZE_STRATEGY]}"
if type(USE_DATA_AUGMENTATION) is not bool:
    raise ValueError("USE_DATA_AUGMENTATION must be True or False")
OUTPUT_DIR = OUTPUT_DIR / ("with_augmentation" if USE_DATA_AUGMENTATION else "no_augmentation")
TARGET_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 5
NUM_EPOCHS = 30
LR = 1e-4  # 1e-3 3e-4 1e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42
N_FOLDS = 5
NUM_WORKERS = 0  # Same as the original notebook.
SAVE_BEST_MODELS = True

# False: original = entire RGB image resized to 224x224, without segmentation.
# True: original keeps its background but uses the SAME mask-guided crop as
# black/imagenet (equivalent to process_one_bg2 in preprocess.ipynb).
# Use True when the question is specifically the effect of background replacement.
ORIGINAL_USE_MASK_CROP = True

# Seed Python, NumPy, Torch and CUDA identically at the beginning of each
# experiment. Deterministic kernels reduce variation between comparisons.
DETERMINISTIC = True

import copy
import csv
import json
import os
import random
import time

# Required by deterministic CUDA matrix multiplication; set before importing torch.
if DETERMINISTIC:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN_COLOR = np.array([124, 116, 104], dtype=np.uint8)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if DETERMINISTIC:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def generate_mask(img_rgb):
    """Original notebook's ExG + Otsu + morphology + component filtering."""
    img_f = img_rgb.astype(np.float32) / 255.0
    R, G, B = img_f[..., 0], img_f[..., 1], img_f[..., 2]
    exg = 2.0 * G - R - B
    exg_norm = (
        (exg - exg.min()) / (exg.max() - exg.min() + 1e-8) * 255
    ).astype(np.uint8)
    _, mask = cv2.threshold(
        exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    h, w = img_rgb.shape[:2]
    iters = 2 if min(h, w) >= 100 else 1
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num > 1:
        min_area = max(5, mask.size * 0.001)
        keep = np.zeros_like(mask)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == i] = 255
        mask = keep
    return mask


def crop_square_resize(img_rgb, mask, target_size=224, pad_ratio=0.10):
    """Reproduce the notebook's crop exactly, including its boundary handling.

    This adjusts a crop inside the image; it does not pad a new square canvas.
    """
    H, W = img_rgb.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        s = min(H, W)
        y1, x1 = (H - s) // 2, (W - s) // 2
        img_rgb = img_rgb[y1:y1 + s, x1:x1 + s]
        mask = mask[y1:y1 + s, x1:x1 + s]
    else:
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        pad = int(max(y2 - y1, x2 - x1) * pad_ratio)
        y1, y2 = max(0, y1 - pad), min(H, y2 + pad)
        x1, x2 = max(0, x1 - pad), min(W, x2 + pad)
        side = max(y2 - y1, x2 - x1)
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        ny1, nx1 = max(0, cy - side // 2), max(0, cx - side // 2)
        ny2, nx2 = min(H, ny1 + side), min(W, nx1 + side)
        if ny2 - ny1 < side:
            ny1 = max(0, ny2 - side)
        if nx2 - nx1 < side:
            nx1 = max(0, nx2 - side)
        img_rgb = img_rgb[ny1:ny2, nx1:nx2]
        mask = mask[ny1:ny2, nx1:nx2]
    img_r = cv2.resize(
        img_rgb, (target_size, target_size), interpolation=cv2.INTER_CUBIC
    )
    msk_r = cv2.resize(
        mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST
    )
    return img_r, msk_r


def collect_samples(root):
    if not root.is_dir():
        raise FileNotFoundError(f"Edit TRAIN_DIR at the top of this file: {root}")
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    if len(classes) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} class folders, found {classes}")
    samples = []
    for label, name in enumerate(classes):
        files = sorted(
            p for p in (root / name).iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
        if len(files) < N_FOLDS:
            raise ValueError(f"Class {name!r} has fewer than {N_FOLDS} images")
        print(f"  {name}: {len(files)}")
        samples.extend((p, label) for p in files)
    return classes, samples


def write_csv(path, rows, fieldnames=None):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def preprocess_samples(samples):
    """Precompute selected variants in RAM; regenerate every run (no stale cache).

    About 72 MiB for 500 x 224 x 224 RGB uint8 images (one version).
    Fail on unreadable samples so all variants retain exactly the same images.
    """
    images = {mode: [] for mode in EXPERIMENTS}
    empty_masks = 0
    need_mask = ORIGINAL_USE_MASK_CROP or any(m != "original" for m in EXPERIMENTS)
    for i, (path, _) in enumerate(samples):
        with Image.open(path) as im:
            img = np.array(ImageOps.exif_transpose(im).convert("RGB"))
        mask = generate_mask(img) if need_mask else None
        if mask is not None and not np.any(mask):
            empty_masks += 1
        for mode in EXPERIMENTS:
            if mode == "original" and not ORIGINAL_USE_MASK_CROP:
                arr = cv2.resize(
                    img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_CUBIC
                )
            else:
                arr = img.copy()
                if mode == "black":
                    arr[mask == 0] = 0
                elif mode == "imagenet":
                    arr[mask == 0] = IMAGENET_MEAN_COLOR
                arr, _ = crop_square_resize(arr, mask, TARGET_SIZE)
            images[mode].append(arr)
        if (i + 1) % 100 == 0 or i + 1 == len(samples):
            print(f"Preprocessed {i + 1}/{len(samples)} images", flush=True)
    print(f"Empty masks: {empty_masks} (same fallback as original notebook)")
    # Save one aligned preview per class, with selected modes side by side.
    preview_dir = OUTPUT_DIR / "previews"
    preview_dir.mkdir(exist_ok=True)
    seen = set()
    for i, (_, label) in enumerate(samples):
        if label not in seen:
            row = np.concatenate([images[m][i] for m in EXPERIMENTS], axis=1)
            Image.fromarray(row).save(preview_dir / f"class_{label}.png")
            seen.add(label)
    return images, empty_masks


class PlantSubset(Dataset):
    def __init__(self, images, labels, indices, transform):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return self.transform(self.images[i]), int(self.labels[i])


def build_resnet18():
    if IMAGENET_WEIGHTS_PATH is None:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
        model.load_state_dict(torch.load(
            IMAGENET_WEIGHTS_PATH, map_location="cpu", weights_only=True
        ))
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    # Only parameter gradients are controlled here. Keep BN in training mode
    # during training for EVERY strategy so running_mean/running_var update.
    for p in model.parameters():
        p.requires_grad = FREEZE_STRATEGY == 3
    if FREEZE_STRATEGY == 2:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True
    return model


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    # All strategies: train -> BN batch stats + running-stat updates;
    # validation -> saved running stats, with no running-stat updates.
    model.train(training)
    running_loss = 0.0
    predictions, targets = [], []
    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            predictions.extend(outputs.argmax(dim=1).detach().cpu().tolist())
            targets.extend(labels.cpu().tolist())
    score = f1_score(targets, predictions, average="macro")
    return running_loss / len(loader.dataset), score, predictions


def build_transforms():
    train_steps = []
    if USE_DATA_AUGMENTATION:
        train_steps.extend([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
        ])
    train_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_transform = transforms.Compose(train_steps)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def run_cv(mode, images, labels, splits, samples, classes, device):
    seed_everything(RANDOM_SEED)
    run_dir = OUTPUT_DIR
    run_dir.mkdir(exist_ok=True)
    train_transform, val_transform = build_transforms()
    criterion = nn.CrossEntropyLoss()
    history, fold_results = [], []
    oof_predictions = np.full(len(labels), -1, dtype=np.int64)
    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        print(f"\n{'=' * 20} {mode}: Fold {fold}/{N_FOLDS} {'=' * 20}")
        print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")
        train_loader = DataLoader(
            PlantSubset(images, labels, train_idx, train_transform),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        )
        val_loader = DataLoader(
            PlantSubset(images, labels, val_idx, val_transform),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        )
        model = build_resnet18().to(device)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Strategy {FREEZE_STRATEGY} ({FREEZE_STRATEGY_NAMES[FREEZE_STRATEGY]}): "
              f"{trainable:,}/{total:,} trainable parameters; BN running stats update during training")
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR, weight_decay=WEIGHT_DECAY,
        )
        best_f1, best_epoch, best_state, best_preds = -1.0, 0, None, None
        for epoch in range(1, NUM_EPOCHS + 1):
            start = time.time()
            train_loss, train_f1, _ = run_epoch(
                model, train_loader, criterion, device, optimizer
            )
            val_loss, val_f1, preds = run_epoch(model, val_loader, criterion, device)
            elapsed = time.time() - start
            history.append(dict(
                experiment=mode, fold=fold, epoch=epoch,
                train_loss=train_loss, train_f1=train_f1,
                val_loss=val_loss, val_f1=val_f1, seconds=elapsed,
            ))
            print(
                f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} | Time: {elapsed:.1f}s",
                flush=True,
            )
            if val_f1 > best_f1:
                best_f1, best_epoch = val_f1, epoch
                best_preds = list(preds)
                if SAVE_BEST_MODELS:
                    best_state = copy.deepcopy(model.state_dict())
                print(f"  -> Best model (Val F1: {best_f1:.4f})")
        if SAVE_BEST_MODELS:
            torch.save(best_state, run_dir / f"best_resnet18_fold{fold}.pth")
        oof_predictions[val_idx] = best_preds
        fold_results.append(dict(
            experiment=mode, fold=fold, best_epoch=best_epoch,
            best_val_f1=best_f1, train_size=len(train_idx), val_size=len(val_idx),
        ))
        write_csv(run_dir / "history.csv", history)
        write_csv(run_dir / "fold_results.csv", fold_results)
        del model, optimizer, best_state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(run_dir / "oof_predictions.csv", [
        dict(sample_index=i, path=str(path.relative_to(TRAIN_DIR)),
             true_label=int(labels[i]), true_class=classes[labels[i]],
             predicted_label=int(oof_predictions[i]),
             predicted_class=classes[oof_predictions[i]])
        for i, (path, _) in enumerate(samples)
    ])
    cm = confusion_matrix(labels, oof_predictions, labels=list(range(NUM_CLASSES)))
    write_csv(run_dir / "confusion_matrix.csv", [
        dict(true_class=name, **{classes[j]: int(cm[i, j]) for j in range(NUM_CLASSES)})
        for i, name in enumerate(classes)
    ])
    scores = [row["best_val_f1"] for row in fold_results]
    result = dict(
        experiment=mode, mean_best_val_f1=float(np.mean(scores)),
        std_best_val_f1=float(np.std(scores)),  # ddof=0, same as notebook
        oof_macro_f1=float(f1_score(labels, oof_predictions, average="macro")),
        **{f"fold_{i + 1}_f1": score for i, score in enumerate(scores)},
    )
    print(f"\n{mode}: Mean {result['mean_best_val_f1']:.4f} "
          f"Std {result['std_best_val_f1']:.4f}")
    return result


def main():
    if not EXPERIMENTS or len(set(EXPERIMENTS)) != len(EXPERIMENTS):
        raise ValueError("EXPERIMENTS must contain unique experiment names")
    if set(EXPERIMENTS) - {"original", "black", "imagenet"}:
        raise ValueError("Valid experiments: original, black, imagenet")
    if NUM_EPOCHS < 1:
        raise ValueError("NUM_EPOCHS must be positive")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    classes, samples = collect_samples(TRAIN_DIR)
    labels = np.array([label for _, label in samples])
    splits = list(StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED
    ).split(np.zeros(len(labels)), labels))
    fold_ids = np.zeros(len(samples), dtype=int)
    for fold, (_, val_idx) in enumerate(splits, 1):
        fold_ids[val_idx] = fold
    write_csv(OUTPUT_DIR / "fold_assignments.csv", [
        dict(sample_index=i, path=str(path.relative_to(TRAIN_DIR)),
             label=label, class_name=classes[label], validation_fold=int(fold_ids[i]))
        for i, (path, label) in enumerate(samples)
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; experiments: {EXPERIMENTS}")
    print(f"Freeze strategy: {FREEZE_STRATEGY} ({FREEZE_STRATEGY_NAMES[FREEZE_STRATEGY]})")
    print(f"Training data augmentation: {USE_DATA_AUGMENTATION}")
    print(f"Original uses mask-guided crop: {ORIGINAL_USE_MASK_CROP}")
    config = dict(
        train_dir=str(TRAIN_DIR), output_dir=str(OUTPUT_DIR),
        imagenet_weights_path=str(IMAGENET_WEIGHTS_PATH), experiments=EXPERIMENTS,
        data_version=DATA_VERSION,
        freeze_strategy=FREEZE_STRATEGY,
        freeze_strategy_name=FREEZE_STRATEGY_NAMES[FREEZE_STRATEGY],
        bn_running_stats_update_during_training=True,
        use_data_augmentation=USE_DATA_AUGMENTATION,
        original_use_mask_crop=ORIGINAL_USE_MASK_CROP, target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE, num_classes=NUM_CLASSES, num_epochs=NUM_EPOCHS,
        lr=LR, weight_decay=WEIGHT_DECAY, random_seed=RANDOM_SEED,
        n_folds=N_FOLDS, num_workers=NUM_WORKERS, deterministic=DETERMINISTIC,
        save_best_models=SAVE_BEST_MODELS, classes=classes, device=str(device),
        torch_version=torch.__version__, preprocessing="ExG/Otsu/open/close/CC",
        preview_columns=list(EXPERIMENTS),
        note="Best epochs selected on validation folds; CV is not independent test performance.",
    )
    images, empty_masks = preprocess_samples(samples)
    config["empty_masks"] = empty_masks
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = []
    for mode in EXPERIMENTS:
        summary.append(run_cv(
            mode, images[mode], labels, splits, samples, classes, device
        ))
        write_csv(OUTPUT_DIR / "comparison_summary.csv", summary)
    print("\n=== Comparison: best validation Macro-F1 (mean +/- population std) ===")
    for row in summary:
        print(f"{row['experiment']:<12} "
              f"{row['mean_best_val_f1']:.4f} +/- {row['std_best_val_f1']:.4f}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
