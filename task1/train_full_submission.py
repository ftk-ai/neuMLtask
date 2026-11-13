"""Kaggle: original neu-mltask1 full-training configuration + submission.

Standalone script; no companion .py file or CV result upload is required.
Run: %run /kaggle/working/train_full_submission.py
"""
from pathlib import Path

# ==================== Kaggle paths: edit here ====================
TRAIN_DIR = Path("/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2/train")
TEST_DIR = Path("/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2/test")
SUBMISSION_TEMPLATE = Path("/kaggle/input/competitions/2026task1/submission-for-task2.csv")
OUTPUT_DIR = Path("/kaggle/working/plant_full_black_original_config")
# None downloads torchvision ImageNet V1 weights (Kaggle Internet required).
# Alternatively, point to a local torchvision ResNet18 ImageNet state_dict.
IMAGENET_WEIGHTS_PATH = None

# Match the actual full-training cell in neu-mltask1.ipynb (not its epoch comments).
# Keep the black-background input preprocessing requested for this script.
NUM_EPOCHS = 22
TARGET_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42
NUM_WORKERS = 0
DETERMINISTIC = False  # The original notebook did not force deterministic kernels.

import csv
import json
import os
import random
import time

if DETERMINISTIC:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import f1_score

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXPECTED_CLASSES = ["Black-grass", "Common wheat", "Loose Silky-bent", "Scentless Mayweed", "Sugar beet"]


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


def preprocess_black(path):
    with Image.open(path) as im:
        img = np.array(ImageOps.exif_transpose(im).convert("RGB"))
    mask = generate_mask(img)
    img[mask == 0] = 0
    result, _ = crop_square_resize(img, mask, TARGET_SIZE)
    return result, not bool(np.any(mask))


def collect_training_samples():
    if not TRAIN_DIR.is_dir():
        raise FileNotFoundError(f"Check TRAIN_DIR: {TRAIN_DIR}")
    classes = sorted(d.name for d in TRAIN_DIR.iterdir() if d.is_dir())
    if classes != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected class folders: {classes}")
    samples = []
    for label, name in enumerate(classes):
        files = sorted(p for p in (TRAIN_DIR / name).iterdir()
                       if p.is_file() and p.suffix.lower() in IMG_EXTS)
        if not files:
            raise ValueError(f"No images in class: {name}")
        print(f"{name}: {len(files)} images")
        samples.extend((p, label) for p in files)
    return classes, samples


def read_submission_template():
    with SUBMISSION_TEMPLATE.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    if fields != ["ID", "Category"] or not rows:
        raise ValueError("Expected a nonempty submission template with ID,Category columns")
    ids = [r["ID"] for r in rows]
    if any(not x.strip() for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("Submission template contains blank or duplicate IDs")
    # Use template order, not directory order. Do not use template categories.
    paths = [TEST_DIR / image_id for image_id in ids]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} test images: {missing[:5]}")
    return fields, rows, paths


def preprocess_many(paths, name):
    images, empty_count = [], 0
    for i, path in enumerate(paths, 1):
        image, empty = preprocess_black(path)
        images.append(image)
        empty_count += int(empty)
        if i % 100 == 0 or i == len(paths):
            print(f"{name}: preprocessed {i}/{len(paths)}", flush=True)
    return images, empty_count


class PlantImages(Dataset):
    def __init__(self, images, transform, labels=None):
        self.images, self.transform, self.labels = images, transform, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.transform(self.images[index])
        return image if self.labels is None else (image, int(self.labels[index]))


def build_transforms():
    train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train, test


def build_model():
    if IMAGENET_WEIGHTS_PATH is None:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
        model.load_state_dict(torch.load(IMAGENET_WEIGHTS_PATH, map_location="cpu", weights_only=True))
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    for p in model.parameters():
        p.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            for p in module.parameters():
                p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()  # FC + BN affine parameters train; BN running stats update.
    total_loss, predictions, targets = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        predictions.extend(outputs.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.cpu().tolist())
    return total_loss / len(loader.dataset), f1_score(targets, predictions, average="macro")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()  # Use saved BN running statistics; never update on test images.
    predictions = []
    for images in loader:
        predictions.extend(model(images.to(device)).argmax(dim=1).cpu().tolist())
    return predictions


def write_submission(path, fields, rows, predictions, classes):
    if len(predictions) != len(rows):
        raise ValueError("Prediction count differs from submission template")
    output_rows = []
    for row, label in zip(rows, predictions):
        if type(label) is not int or not 0 <= label < len(classes):
            raise ValueError(f"Invalid prediction: {label}")
        output_rows.append({"ID": row["ID"], "Category": classes[label]})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        saved = list(reader)
        assert reader.fieldnames == fields
    assert len(saved) == len(rows) and [r["ID"] for r in saved] == [r["ID"] for r in rows]
    assert all(r["Category"] in classes for r in saved)


def main():
    if type(NUM_EPOCHS) is not int or NUM_EPOCHS < 1:
        raise ValueError("NUM_EPOCHS must be a positive integer")
    epochs = NUM_EPOCHS
    classes, samples = collect_training_samples()
    fields, rows, test_paths = read_submission_template()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Original notebook training configuration: FC + BN, LR={LR}, epochs={epochs}")
    train_images, train_empty = preprocess_many([p for p, _ in samples], "train")
    test_images, test_empty = preprocess_many(test_paths, "test")
    train_transform, test_transform = build_transforms()
    labels = [label for _, label in samples]
    train_loader = DataLoader(PlantImages(train_images, train_transform, labels),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(PlantImages(test_images, test_transform),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    seed_everything(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    config = dict(
        train_dir=str(TRAIN_DIR), test_dir=str(TEST_DIR), submission_template=str(SUBMISSION_TEMPLATE),
        imagenet_weights_path=None if IMAGENET_WEIGHTS_PATH is None else str(IMAGENET_WEIGHTS_PATH),
        classes=classes, class_to_idx={name: i for i, name in enumerate(classes)},
        data_version="black", freeze_strategy=2, freeze_strategy_name="fc_bn", use_data_augmentation=True,
        num_epochs=epochs, epoch_selection="fixed 22, as executed in neu-mltask1.ipynb full-training cell",
        training_config_source="neu-mltask1.ipynb cell 2",
        lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE, target_size=TARGET_SIZE,
        random_seed=RANDOM_SEED, num_workers=NUM_WORKERS, deterministic=DETERMINISTIC,
        train_samples=len(samples), test_samples=len(rows), device=str(device),
        empty_masks_train=train_empty, empty_masks_test=test_empty,
        torch_version=torch.__version__, bn_running_stats_update_during_training=True,
        test_bn_mode="eval", model_selection="final epoch; no validation/test-based selection",
    )
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    history = []
    for epoch in range(1, epochs + 1):
        start = time.time()
        loss, score = train_one_epoch(model, train_loader, criterion, optimizer, device)
        elapsed = time.time() - start
        history.append(dict(epoch=epoch, train_loss=loss, train_f1=score, seconds=elapsed))
        with (OUTPUT_DIR / "history.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(f"Epoch {epoch:02d}/{epochs} | Loss: {loss:.4f} | Train Macro-F1: {score:.4f} | {elapsed:.1f}s", flush=True)
        # Same final-five-epoch checkpoints as the original notebook.
        if epoch >= epochs - 4:
            torch.save(model.state_dict(), OUTPUT_DIR / f"resnet18_plant_full_epoch{epoch}.pth")
    model_path = OUTPUT_DIR / "resnet18_plant_full_final.pth"
    torch.save(model.state_dict(), model_path)
    predictions = predict(model, test_loader, device)
    submission_path = OUTPUT_DIR / "submission.csv"
    write_submission(submission_path, fields, rows, predictions, classes)
    print("Test prediction counts:")
    for i, name in enumerate(classes):
        print(f"  {name}: {predictions.count(i)}")
    print(f"Saved model: {model_path}")
    print(f"Saved submission ({len(rows)} rows): {submission_path}")


if __name__ == "__main__":
    main()
