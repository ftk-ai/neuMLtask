"""Kaggle: CVLFace AdaFace ViT -> 当前比赛的六类人脸表情。

先修改下方路径，再运行：python train_cvlface_vit_fer.py
Kaggle Notebook 可用：!python /kaggle/working/train_cvlface_vit_fer.py
首次下载权重需开启 Internet 和 GPU。依赖安装在 Notebook 中单独运行：
%pip install -q timm==0.9.16 omegaconf==2.3.0 fvcore==0.1.5.post20221221 huggingface_hub safetensors scikit-learn pandas matplotlib Pillow tqdm
保留 Kaggle 自带的 torch/torchvision，不要随意升级或降级。
这是六分类比赛版本；标准七分类 FER2013 需调整 CLASS_NAMES 和提交映射。
"""

from pathlib import Path

# ==================== 路径：上传 Kaggle 后优先修改这里 ====================
# 指向包含 train/test 的目录，或其上面的 fer_data / 2026task2 层。
DATA_ROOT = Path("/kaggle/input/competitions/2026task2/fer_data/fer_data")
SUBMISSION_TEMPLATE = Path("/kaggle/input/competitions/2026task2/submission.csv")
OUTPUT_DIR = Path("/kaggle/working/cvlface_vit_fer")
# 推荐上传已有的 split.csv 并填写路径；None 时固定 seed=42 分层划分。
# 两个脚本必须指向同一个 split.csv，或都使用相同数据和默认划分配置。
SPLIT_PATH = None  # 例：Path("/kaggle/input/your-split/split.csv")
# 离线模型目录：包含 models/、pretrained_model/model.yaml、model.safetensors。
LOCAL_MODEL_DIR = None  # 例：Path("/kaggle/input/cvlface-model")
MODEL_CACHE_DIR = Path("/kaggle/working/cvlface_pretrained")
# 仅预测时设 INFERENCE_ONLY=True，并指向本脚本保存的 best.pt。
CHECKPOINT_PATH = None  # 例：Path("/kaggle/input/your-checkpoint/best.pt")

# ==================== 实验配置 ====================
MODEL_ID = "minchul/cvlface_adaface_vit_base_webface4m"
MODEL_REVISION = "c6fed2ff09ac0a29cccc72c09127ad7b4bd3fa9e"
IMAGE_SIZE = 112
FEATURE_DIM = 512
SEED = 42
VAL_RATIO = 0.20
BATCH_SIZE = 16
ACCUM_STEPS = 2  # 有效 batch 约为 32；显存不足可改为 8、4。
NUM_WORKERS = 2
HEAD_EPOCHS = 3
FINETUNE_EPOCHS = 15
HEAD_LR = 1e-3
BACKBONE_LR = 1e-5
FINETUNE_HEAD_LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5
LABEL_SMOOTHING = 0.05
USE_CLASS_WEIGHTS = True
USE_AMP = True
USE_FLIP_TTA = True
FREEZE_BN_STATS = True
INFERENCE_ONLY = False
SMOKE_TEST = False  # 小样本真实模型试跑；仍需下载权重，不生成正式 submission。
AUDIT_IMAGES = True  # 正式训练前检查全部图像是否可解码。

CLASS_NAMES = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
# 沿用项目已有的数字提交编码，注意与内部类别顺序不同。
SUBMISSION_LABEL_MAP = {"Angry": 0, "Fear": 1, "Happy": 2,
                        "Sad": 3, "Surprise": 4, "Neutral": 5}

import importlib.util
import json
import random
import sys
from contextlib import nullcontext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def locate_image_root(root):
    candidates = [root, root / "fer_data", root / "fer_data" / "fer_data"]
    matches = [p for p in candidates if (p / "train").is_dir() and (p / "test").is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(f"无法唯一定位 train/test，请修改 DATA_ROOT：{candidates}")
    return matches[0]


def validate_fixed_split(rows, expected):
    """纯 Python 校验：完整覆盖、无重复/交集、标签正确，并保留 CSV 内顺序。"""
    groups = {"train": [], "val": []}
    seen = set()
    for row in rows:
        name = row["relative_path"]
        if name in seen or name not in expected:
            raise ValueError(f"split.csv 图片重复或不存在：{name}")
        if int(row["label"]) != expected[name]:
            raise ValueError(f"split.csv 标签与类别文件夹不一致：{name}")
        if row["split"] not in groups:
            raise ValueError(f"未知 split：{row['split']}")
        groups[row["split"]].append(name)
        seen.add(name)
    if seen != set(expected) or not all(groups.values()):
        raise ValueError("split.csv 必须覆盖全部有标签图片，且 train/val 均非空。")
    return groups


def validate_template_ids(ids, image_names):
    if not ids or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("提交 ID 必须为非空文件名。")
    if len(ids) != len(set(ids)):
        raise ValueError("提交模板中存在重复 ID。")
    if any(x in {".", ".."} or "/" in x or "\\" in x or ":" in x for x in ids):
        raise ValueError("提交 ID 必须是直接位于 test 下的文件名。")
    if set(ids) != set(image_names):
        raise ValueError("提交模板 ID 与 test 图片不完全对应。")


def prepare_data(out):
    root = locate_image_root(DATA_ROOT)
    train_dir, test_dir = root / "train", root / "test"
    actual = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    if actual != sorted(CLASS_NAMES):
        raise ValueError(f"需要六类 {CLASS_NAMES}，实际为 {actual}")
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    records = []
    for name in CLASS_NAMES:
        paths = sorted(p for p in (train_dir / name).rglob("*")
                       if p.is_file() and p.suffix.lower() in suffixes)
        for path in paths:
            records.append(dict(path=str(path), relative_path=path.relative_to(train_dir).as_posix(),
                                class_name=name, label=CLASS_TO_IDX[name]))
    frame = pd.DataFrame(records)
    if frame.empty or set(frame["label"]) != set(range(len(CLASS_NAMES))):
        raise ValueError("训练集为空或存在空类别。")
    if SPLIT_PATH is not None:
        split = pd.read_csv(SPLIT_PATH)
        if not {"relative_path", "label", "split"}.issubset(split.columns):
            raise ValueError("split.csv 缺少 relative_path、label 或 split 列。")
        groups = validate_fixed_split(split.to_dict("records"),
                                      dict(zip(frame["relative_path"], frame["label"])))
        indexed = frame.set_index("relative_path", drop=False)
        train_df = indexed.loc[groups["train"]].reset_index(drop=True)
        val_df = indexed.loc[groups["val"]].reset_index(drop=True)
    else:
        train_df, val_df = train_test_split(frame, test_size=VAL_RATIO, random_state=SEED,
                                          stratify=frame["label"])
        train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)
    for name, part in [("train", train_df), ("val", val_df)]:
        if set(part["label"]) != set(range(len(CLASS_NAMES))):
            raise ValueError(f"{name} 未覆盖全部类别。")
    print("类别数量：\n", frame.groupby("class_name").size().reindex(CLASS_NAMES))
    print(f"train={len(train_df)}, val={len(val_df)}")
    pd.concat([train_df.assign(split="train"), val_df.assign(split="val")])[
        ["relative_path", "label", "split"]].to_csv(out / "split.csv", index=False)

    template = pd.read_csv(SUBMISSION_TEMPLATE, dtype={"ID": str})
    if list(template.columns) != ["ID", "Emotion"]:
        raise ValueError("提交模板必须包含且仅包含 ID,Emotion 两列。")
    test_paths = sorted(p for p in test_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in suffixes)
    validate_template_ids(template["ID"].tolist(), [p.name for p in test_paths])
    test_df = pd.DataFrame({"ID": template["ID"],
                            "path": [str(test_dir / x) for x in template["ID"]]})
    if AUDIT_IMAGES:
        sizes = {}
        for path in tqdm(frame["path"].tolist() + test_df["path"].tolist(), desc="检查图像"):
            with Image.open(path) as image:
                image.load()
                key = str((image.size, image.mode))
                sizes[key] = sizes.get(key, 0) + 1
        print("图像尺寸/模式：", sizes)
    if SMOKE_TEST:
        train_df = train_df.groupby("label", group_keys=False).head(8).reset_index(drop=True)
        val_df = val_df.groupby("label", group_keys=False).head(4).reset_index(drop=True)
        test_df = test_df.head(16).reset_index(drop=True)
        template = template.head(16).reset_index(drop=True)
    # 保存实际使用的清单；烟雾模式输出在独立 smoke 子目录。
    train_df.to_csv(out / "train_split.csv", index=False)
    val_df.to_csv(out / "val_split.csv", index=False)
    return train_df, val_df, test_df, template


def build_transforms():
    normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)
    resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC)
    train = transforms.Compose([
        resize, transforms.RandomHorizontalFlip(0.5),
        transforms.RandomAffine(8, translate=(0.04, 0.04),
                                interpolation=InterpolationMode.BILINEAR, fill=128),
        transforms.ToTensor(), normalize,
    ])
    evaluate = transforms.Compose([resize, transforms.ToTensor(), normalize])
    return train, evaluate


class EmotionDataset(Dataset):
    def __init__(self, frame, transform, labeled=True):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.labeled = labeled

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("L").convert("RGB"))
        target = int(row["label"]) if self.labeled else str(row["ID"])
        return tensor, target


def make_loader(frame, transform, device, training=False, labeled=True):
    return DataLoader(
        EmotionDataset(frame, transform, labeled), batch_size=BATCH_SIZE,
        shuffle=training, drop_last=training, num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(SEED if training else SEED + 1),
        persistent_workers=NUM_WORKERS > 0,
    )


def build_backbone(load_pretrained=True):
    # 固定 revision，使用与原 notebook 相同的官方源码和 safetensors。
    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    if LOCAL_MODEL_DIR is not None:
        root = Path(LOCAL_MODEL_DIR).expanduser().resolve()
    else:
        patterns = ["models/**", "pretrained_model/model.yaml", "config.json"]
        if load_pretrained:
            patterns.append("model.safetensors")
        root = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION,
                                     local_dir=str(MODEL_CACHE_DIR), allow_patterns=patterns)).resolve()
    required = ["models/__init__.py", "models/vit/vit.py", "pretrained_model/model.yaml"]
    if load_pretrained:
        required.append("model.safetensors")
    missing = [x for x in required if not (root / x).is_file()]
    if missing:
        raise FileNotFoundError(f"模型目录 {root} 缺少 {missing}")
    name = "_cvlface_adaface_official"
    spec = importlib.util.spec_from_file_location(name, root / "models/__init__.py",
                                                 submodule_search_locations=[str(root / "models")])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    config = OmegaConf.load(root / "pretrained_model/model.yaml")
    config.start_from = ""
    config.freeze = False
    backbone = module.get_model(config)
    if load_pretrained:
        state = load_file(str(root / "model.safetensors"), device="cpu")
        if state and all(k.startswith("model.") for k in state):
            state = {k[len("model."):]: v for k, v in state.items()}
        print("严格加载预训练骨干：", backbone.load_state_dict(state, strict=True))
    return backbone


class EmotionModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.LayerNorm(FEATURE_DIM), nn.Dropout(0.3),
                                  nn.Linear(FEATURE_DIM, len(CLASS_NAMES)))
        self.backbone_frozen = False

    def forward(self, x):
        with torch.no_grad() if self.backbone_frozen else nullcontext():
            features = self.backbone(x)
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise RuntimeError(f"骨干输出形状异常：{features.shape}")
        return self.head(features)

    def set_frozen(self, frozen):
        self.backbone_frozen = frozen
        self.backbone.requires_grad_(not frozen)

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        elif mode and FREEZE_BN_STATS:
            for layer in self.backbone.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    layer.eval()
        return self


def amp_context(device):
    return torch.autocast("cuda", dtype=torch.float16) if USE_AMP and device.type == "cuda" else nullcontext()


def run_epoch(model, loader, criterion, device, scaler, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, seen = 0.0, 0
    targets, predictions = [], []
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for step, (images, labels) in enumerate(tqdm(loader, leave=False)):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with amp_context(device):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("loss 非有限；请检查输入、学习率或关闭 AMP。")
            if training:
                group_start = (step // ACCUM_STEPS) * ACCUM_STEPS
                group_size = min(ACCUM_STEPS, len(loader) - group_start)
                scaler.scale(loss / group_size).backward()
                if (step + 1) % ACCUM_STEPS == 0 or step + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.item() * labels.size(0)
            seen += labels.size(0)
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.detach().argmax(1).cpu().tolist())
    if not seen:
        raise ValueError("DataLoader 为空；请减小 BATCH_SIZE。")
    # 带类别权重时，这是 batch loss 的样本加权平均，用于监控趋势。
    return dict(loss=loss_sum / seen, accuracy=accuracy_score(targets, predictions),
                macro_f1=f1_score(targets, predictions, labels=list(range(len(CLASS_NAMES))),
                                  average="macro", zero_division=0))


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["class_names"] != CLASS_NAMES:
        raise ValueError("权重类别顺序与当前配置不一致。")
    if checkpoint["config"]["MODEL_ID"] != MODEL_ID:
        raise ValueError("权重所属模型与当前脚本不一致。")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return checkpoint


def fit(model, train_loader, val_loader, train_df, device, out, config):
    counts = np.bincount(train_df["label"].to_numpy(), minlength=len(CLASS_NAMES))
    weights = torch.tensor(len(train_df) / (len(CLASS_NAMES) * counts),
                           dtype=torch.float32, device=device) if USE_CLASS_WEIGHTS else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)
    enabled = USE_AMP and device.type == "cuda"
    scaler = (torch.amp.GradScaler("cuda", enabled=enabled)
              if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=enabled))
    best_path = out / "best.pt"
    history, best_score = [], -float("inf")
    stages = [("head", 1 if SMOKE_TEST else HEAD_EPOCHS, True),
              ("finetune", 1 if SMOKE_TEST else FINETUNE_EPOCHS, False)]
    for stage, epochs, frozen in stages:
        if stage == "finetune":
            load_checkpoint(model, best_path)  # 阶段一最优权重作为阶段二起点。
        model.set_frozen(frozen)
        groups = [{"params": model.head.parameters(), "lr": HEAD_LR}] if frozen else [
            {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
            {"params": model.head.parameters(), "lr": FINETUNE_HEAD_LR}]
        optimizer = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        stage_best, stale = -float("inf"), 0
        for epoch in range(1, epochs + 1):
            train_metrics = run_epoch(model, train_loader, criterion, device, scaler, optimizer)
            val_metrics = run_epoch(model, val_loader, criterion, device, scaler)
            row = dict(stage=stage, epoch=epoch, global_epoch=len(history) + 1,
                       backbone_lr=0.0 if frozen else optimizer.param_groups[0]["lr"],
                       head_lr=optimizer.param_groups[-1]["lr"])
            row.update({f"train_{k}": v for k, v in train_metrics.items()})
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            history.append(row)
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
            score = val_metrics["macro_f1"]
            print(f"{stage} {epoch}/{epochs} | train_loss={train_metrics['loss']:.4f} | "
                  f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
                  f"val_macro_f1={score:.4f}", flush=True)
            if score > best_score:
                best_score = score
                torch.save(dict(model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                class_names=CLASS_NAMES, class_to_idx=CLASS_TO_IDX, config=config,
                                best_macro_f1=float(score), val_accuracy=float(val_metrics["accuracy"]),
                                stage=stage, epoch=epoch), best_path)
            if score > stage_best:
                stage_best, stale = score, 0
            else:
                stale += 1
            scheduler.step()
            if stale >= PATIENCE:
                print(f"{stage} 早停。")
                break
    model.set_frozen(False)
    checkpoint = load_checkpoint(model, best_path)
    print("最佳模型：", checkpoint["stage"], checkpoint["epoch"],
          "val_macro_f1=", checkpoint["best_macro_f1"])
    history_df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ["loss", "accuracy", "macro_f1"]):
        ax.plot(history_df["global_epoch"], history_df[f"train_{metric}"], label="Train")
        ax.plot(history_df["global_epoch"], history_df[f"val_{metric}"], label="Validation")
        ax.set(xlabel="Epoch", title=metric)
        ax.legend()
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)
    return checkpoint


@torch.inference_mode()
def predict_probabilities(model, loader, device, flip_tta=False):
    model.eval()
    probabilities, targets = [], []
    for images, target in tqdm(loader, desc="预测"):
        images = images.to(device, non_blocking=True)
        with amp_context(device):
            probs = model(images).float().softmax(1)
            if flip_tta:
                probs = (probs + model(images.flip(-1)).float().softmax(1)) / 2
        if not torch.isfinite(probs).all() or not torch.allclose(
                probs.sum(1), torch.ones(probs.size(0), device=device), atol=1e-4):
            raise FloatingPointError("预测概率异常。")
        probabilities.append(probs.cpu().numpy())
        targets.extend(target.tolist() if torch.is_tensor(target) else list(target))
    return np.concatenate(probabilities), targets


def export_results(model, val_loader, test_loader, val_df, template, device, out, checkpoint):
    # 验证不使用 TTA，与训练时选择模型的指标一致。
    val_probs, targets = predict_probabilities(model, val_loader, device)
    predictions = val_probs.argmax(1)
    report = classification_report(targets, predictions, labels=list(range(len(CLASS_NAMES))),
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out / "classification_report.csv")
    metrics = dict(val_accuracy=float(accuracy_score(targets, predictions)),
                   val_macro_f1=float(f1_score(targets, predictions, labels=list(range(len(CLASS_NAMES))),
                                               average="macro", zero_division=0)),
                   val_tta=False, test_flip_tta=USE_FLIP_TTA, smoke_test=SMOKE_TEST,
                   selected_stage=checkpoint["stage"], selected_epoch=checkpoint["epoch"])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cm = confusion_matrix(targets, predictions, labels=list(range(len(CLASS_NAMES))))
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out / "confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(xticks=range(len(CLASS_NAMES)), yticks=range(len(CLASS_NAMES)),
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, xlabel="Predicted", ylabel="True")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    val_output = val_df.copy()
    val_output["predicted_class"] = [CLASS_NAMES[i] for i in predictions]
    val_output.to_csv(out / "val_predictions.csv", index=False)

    test_probs, ids = predict_probabilities(model, test_loader, device, USE_FLIP_TTA)
    if ids != template["ID"].tolist():
        raise RuntimeError("预测 ID 顺序与提交模板不一致。")
    names = [CLASS_NAMES[i] for i in test_probs.argmax(1)]
    output = pd.DataFrame({"ID": ids, "predicted_class": names})
    for i, name in enumerate(CLASS_NAMES):
        output[f"prob_{name}"] = test_probs[:, i]
    output.to_csv(out / "test_predictions.csv", index=False)
    if not SMOKE_TEST:
        submission = template.copy()
        submission["Emotion"] = [SUBMISSION_LABEL_MAP[x] for x in names]
        submission.to_csv(out / "submission.csv", index=False)
        saved = pd.read_csv(out / "submission.csv", dtype={"ID": str})
        if saved["ID"].tolist() != ids or not saved["Emotion"].isin(range(len(CLASS_NAMES))).all():
            raise RuntimeError("提交文件回读校验失败。")
        print("提交文件：", out / "submission.csv", f"({len(saved)} 行)")
    else:
        print("SMOKE_TEST：仅导出小样本预测，不生成正式提交。")
    print("验证指标（不是 Kaggle 测试分数）：", metrics)


def main():
    if BATCH_SIZE < 2 or ACCUM_STEPS < 1 or NUM_WORKERS < 0:
        raise ValueError("要求 BATCH_SIZE>=2、ACCUM_STEPS>=1、NUM_WORKERS>=0。")
    if HEAD_EPOCHS < 1 or FINETUNE_EPOCHS < 0 or PATIENCE < 1 or not 0 < VAL_RATIO < 1:
        raise ValueError("训练轮数、早停参数或 VAL_RATIO 无效。")
    if set(SUBMISSION_LABEL_MAP) != set(CLASS_NAMES) or set(SUBMISSION_LABEL_MAP.values()) != set(range(len(CLASS_NAMES))):
        raise ValueError("提交标签映射必须完整对应 0 到 5。")
    if INFERENCE_ONLY and (CHECKPOINT_PATH is None or not Path(CHECKPOINT_PATH).is_file()):
        raise FileNotFoundError("仅预测模式需要有效的 CHECKPOINT_PATH。")
    if not SUBMISSION_TEMPLATE.is_file():
        raise FileNotFoundError(f"请修改 SUBMISSION_TEMPLATE：{SUBMISSION_TEMPLATE}")
    out = OUTPUT_DIR / "smoke" if SMOKE_TEST else OUTPUT_DIR
    for source in [SUBMISSION_TEMPLATE, SPLIT_PATH]:
        if source is not None and (out / Path(source).name).resolve() == Path(source).resolve():
            raise ValueError("输出目录不能覆盖输入模板或划分文件。")
    if CHECKPOINT_PATH is not None and (out / "best.pt").resolve() == Path(CHECKPOINT_PATH).resolve() and not INFERENCE_ONLY:
        raise ValueError("新训练不能覆盖 CHECKPOINT_PATH。")
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "Torch:", torch.__version__, "Torchvision:", torchvision.__version__)
    if device.type == "cpu":
        print("当前是 CPU；正式训练请在 Kaggle 开启 GPU。")
    train_df, val_df, test_df, template = prepare_data(out)
    train_transform, eval_transform = build_transforms()
    train_loader = make_loader(train_df, train_transform, device, training=True)
    val_loader = make_loader(val_df, eval_transform, device)
    test_loader = make_loader(test_df, eval_transform, device, labeled=False)
    if not len(train_loader):
        raise ValueError("训练集不足一个 batch，请减小 BATCH_SIZE。")
    seed_everything(SEED)
    model = EmotionModel(build_backbone(load_pretrained=not INFERENCE_ONLY)).to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device))
    if logits.shape != (2, len(CLASS_NAMES)) or not torch.isfinite(logits).all():
        raise RuntimeError("模型输出形状或数值异常。")
    print("参数量：", f"{sum(p.numel() for p in model.parameters()):,}")
    (out / "model_architecture.txt").write_text(str(model), encoding="utf-8")
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in globals().items() if k.isupper()}
    config.update(torch_version=str(torch.__version__), torchvision_version=str(torchvision.__version__),
                  actual_train_size=len(train_df), actual_val_size=len(val_df), actual_test_size=len(test_df))
    (out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    checkpoint = (load_checkpoint(model, CHECKPOINT_PATH) if INFERENCE_ONLY else
                  fit(model, train_loader, val_loader, train_df, device, out, config))
    export_results(model, val_loader, test_loader, val_df, template, device, out, checkpoint)


if __name__ == "__main__":
    main()
