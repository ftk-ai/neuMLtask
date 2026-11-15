"""CVLFace：每类 5 张验证图片的 24 层平均接收注意力图。

运行：python task2/visualize_cvlface_received_attention.py
依赖：匹配的 torch/torchvision、numpy、Pillow、matplotlib。
不训练、不下载权重，不依赖 timm/fvcore/Hugging Face。
下方用纯 PyTorch 重建当前 CVLFace ViT Base 的推理结构，并严格加载 best.pt。
结构来源：CVLface 官方 models/vit/vit.py；训练专用 DropPath 在 eval 下为恒等映射。
本图衡量 patch 被全部 Query 平均关注的程度，不是 CLS 图或类别贡献图。
"""

from pathlib import Path

# ==================== 文件路径：优先修改这里 ====================
TASK_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = TASK_DIR / "resultcvlface" / "best.pt"
VAL_SPLIT_PATH = TASK_DIR / "resultcvlface" / "val_split.csv"
# 指向 Angry/Fear/Happy/... 类别文件夹的父目录。
TRAIN_DIR = TASK_DIR / "dataset" / "2026task2" / "fer_data" / "fer_data" / "train"
OUTPUT_DIR = TASK_DIR / "resultcvlface" / "received_attention_visualizations"
# 可选：对照 Kaggle 导出的分类结果，验证重建模型在所选图片上的预测是否一致。
REFERENCE_PREDICTIONS_PATH = TASK_DIR / "resultcvlface" / "val_predictions.csv"  # 可设为 None

# Kaggle 示例：将上面的路径替换为上传文件的实际路径。
# CHECKPOINT_PATH = Path("/kaggle/input/your-results/best.pt")
# VAL_SPLIT_PATH = Path("/kaggle/input/your-results/val_split.csv")
# TRAIN_DIR = Path("/kaggle/input/competitions/2026task2/fer_data/fer_data/train")
# OUTPUT_DIR = Path("/kaggle/working/cvlface_received_attention")
# REFERENCE_PREDICTIONS_PATH = None

# ==================== 可视化配置 ====================
IMAGES_PER_CLASS = 5
SELECTED_RELATIVE_PATHS = []  # 留空：每类取验证清单前 5 张；也可指定验证图片。
DEVICE = "auto"  # "auto"、"cpu"、"cuda"
CPU_THREADS = 4
OVERLAY_ALPHA = 0.45
COLORMAP = "inferno"
DPI = 150
SAVE_ATTENTION_ARRAYS = True
VERIFY_FORWARD = True

import csv
import json
from pathlib import PurePosixPath

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 512, kernel_size=8, stride=8)

    def forward(self, x):
        if tuple(x.shape[1:]) != (3, 112, 112):
            raise ValueError("CVLFace ViT Base 需要 [B,3,112,112] 输入。")
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 16
        self.scale = (512 // self.num_heads) ** -0.5
        self.qkv = nn.Linear(512, 1536, bias=False)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(512, 512)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x):
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, dim // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        # 捕获这个位置的 [B,16,196,196] 权重；hook 不改变前向计算。
        attention = self.attn_drop(attention)
        x = (attention @ v).transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 1536)
        self.act = nn.ReLU6()
        self.fc2 = nn.Linear(1536, 512)
        self.drop = nn.Dropout(0.0)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(512, eps=1e-5)
        self.norm2 = nn.LayerNorm(512, eps=1e-5)
        self.attn = Attention()
        # 仅用于推理：官方 DropPath 在 eval 下也是恒等映射，不包含参数。
        self.drop_path = nn.Identity()
        self.mlp = Mlp()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.pos_embed = nn.Parameter(torch.zeros(1, 196, 512))
        self.pos_drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList([Block() for _ in range(24)])
        self.norm = nn.LayerNorm(512, eps=1e-5)
        self.feature = nn.Sequential(
            nn.Linear(100352, 512, bias=False),
            nn.BatchNorm1d(512, eps=2e-5),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512, eps=2e-5),
        )

    def forward(self, x):
        x = self.pos_drop(self.patch_embed(x) + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x.float())
        return self.feature(x.reshape(x.shape[0], 100352))


class ViTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = VisionTransformer()

    def forward(self, x):
        # 输入为复制后的三通道灰度图；官方 wrapper 中可能的 RGB/BGR 翻转不改变数值。
        return self.net(x)


class EmotionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = ViTModel()
        self.head = nn.Sequential(nn.LayerNorm(512), nn.Dropout(0.3), nn.Linear(512, num_classes))

    def forward(self, x):
        return self.head(self.backbone(x))


def load_model(device):
    checkpoint = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=True, mmap=True)
    config = checkpoint.get("config", {})
    expected = "minchul/cvlface_adaface_vit_base_webface4m"
    if config.get("MODEL_ID") != expected or config.get("IMAGE_SIZE") != 112:
        raise ValueError("需要 train_cvlface_vit_fer.py 保存的 112x112 CVLFace ViT Base 权重。")
    classes = checkpoint["class_names"]
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("权重中的类别名称无效。")
    model = EmotionModel(len(classes))
    # 检查全部参数和 BN 统计量；不忽略 missing/unexpected keys。
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval().requires_grad_(False)
    metadata = {k: checkpoint.get(k) for k in ["stage", "epoch", "best_macro_f1", "val_accuracy"]}
    return model, classes, metadata


def read_selected_rows(classes):
    with VAL_SPLIT_PATH.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not {"relative_path", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("验证清单必须包含 relative_path 和 label。")
        rows = list(reader)
    seen = set()
    for row in rows:
        relative = row["relative_path"].replace("\\", "/")
        parts = PurePosixPath(relative)
        if parts.is_absolute() or ":" in relative or ".." in parts.parts or len(parts.parts) < 2:
            raise ValueError(f"非法相对路径：{relative}")
        if relative in seen:
            raise ValueError(f"验证清单图片重复：{relative}")
        seen.add(relative)
        label = int(row["label"])
        if not 0 <= label < len(classes) or parts.parts[0] != classes[label]:
            raise ValueError(f"路径与类别映射不一致：{relative}")
        if row.get("class_name") and row["class_name"] != classes[label]:
            raise ValueError(f"类别名称与标签不一致：{relative}")
        row.update(relative_path=relative, label=label)
    if SELECTED_RELATIVE_PATHS:
        keys = [x.replace("\\", "/") for x in SELECTED_RELATIVE_PATHS]
        by_path = {x["relative_path"]: x for x in rows}
        if len(keys) != len(set(keys)) or any(x not in by_path for x in keys):
            raise ValueError("指定图片存在重复，或不属于验证集。")
        return [by_path[x] for x in keys]
    if IMAGES_PER_CLASS < 1:
        raise ValueError("IMAGES_PER_CLASS 必须为正整数。")
    groups = [[x for x in rows if x["label"] == i] for i in range(len(classes))]
    if any(len(group) < IMAGES_PER_CLASS for group in groups):
        raise ValueError("至少一个类别的验证图片不足指定数量。")
    return [group[j] for j in range(IMAGES_PER_CLASS) for group in groups]


def received_attention(weights):
    """[heads,queries,keys] -> [keys]：保留被读取的位置，不能对 Key 维平均。"""
    if weights.ndim != 3 or weights.shape[1] != weights.shape[2]:
        raise ValueError("需要 [heads,queries,keys] 方阵。")
    # 先平均头，后平均 Query。每个 Key 得到它被全部 Query 平均关注的权重。
    return weights.float().mean(dim=0).mean(dim=0)


class AttentionCapture:
    def __init__(self, model):
        self.layers = list(model.backbone.net.blocks)
        self.maps = [None] * 24
        self.self_mass = [None] * 24
        self.handles = []

    def record(self, layer):
        def hook(module, inputs, output):
            if tuple(output.shape) != (1, 16, 196, 196):
                raise RuntimeError(f"注意力形状异常：{output.shape}")
            weights = output[0].detach().float()
            if not torch.isfinite(weights).all() or (weights < 0).any():
                raise FloatingPointError("注意力包含非有限值或负值。")
            torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)),
                                       atol=1e-4, rtol=1e-4)
            self.maps[layer] = received_attention(weights).cpu()
            self.self_mass[layer] = float(weights.diagonal(dim1=-2, dim2=-1).mean().item())
        return hook

    def __enter__(self):
        for i, layer in enumerate(self.layers):
            self.handles.append(layer.attn.attn_drop.register_forward_hook(self.record(i)))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()

    def arrays(self):
        if any(x is None for x in self.maps):
            raise RuntimeError("未捕获到全部 24 层。")
        maps = torch.stack(self.maps)
        torch.testing.assert_close(maps.sum(-1), torch.ones(24), atol=1e-4, rtol=1e-4)
        return maps.reshape(24, 14, 14).numpy(), np.asarray(self.self_mass, dtype=np.float32)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figures(image, maps, self_mass, folder, title):
    # 同一图片 24 层共用 [0,max] 色标，保留跨层绝对数值差异。
    vmax = max(float(maps.max()), 1e-12)
    fig, axes = plt.subplots(4, 6, figsize=(19, 13), layout="constrained")
    statistics = []
    for i, ax in enumerate(axes.flat):
        grid = maps[i]
        upsampled = F.interpolate(torch.from_numpy(grid.copy())[None, None], size=(112, 112),
                                  mode="bilinear", align_corners=False)[0, 0].numpy()
        ax.imshow(image)
        ax.imshow(upsampled, cmap=COLORMAP, vmin=0, vmax=vmax, alpha=OVERLAY_ALPHA)
        ax.set_title(f"Layer {i + 1}", fontsize=11)
        ax.axis("off")
        flat = grid.ravel()
        entropy = float(-(flat * np.log(np.maximum(flat, 1e-30))).sum() / np.log(196))
        statistics.append(dict(layer=i + 1, min_weight=float(flat.min()), max_weight=float(flat.max()),
                               mean_weight=float(flat.mean()), sum_weight=float(flat.sum()),
                               normalized_entropy=entropy, mean_self_attention=float(self_mass[i])))
    fig.suptitle(title + "\nMean received attention: mean over 16 heads and 196 queries | shared scale across 24 layers",
                 fontsize=12)
    scalar = plt.cm.ScalarMappable(norm=Normalize(0, vmax), cmap=COLORMAP)
    fig.colorbar(scalar, ax=list(axes.flat), shrink=0.65, label="Mean received attention weight")
    fig.savefig(folder / "received_attention_24_layers.png", dpi=DPI)
    plt.close(fig)
    image.save(folder / "input_112.png")
    write_csv(folder / "layer_statistics.csv", statistics)
    if SAVE_ATTENTION_ARRAYS:
        np.savez_compressed(folder / "received_attention_maps.npz", received_attention=maps,
                            mean_self_attention=self_mass)


def main():
    for path in [CHECKPOINT_PATH, VAL_SPLIT_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f"请修改文件顶部路径：{path}")
    if not TRAIN_DIR.is_dir():
        raise FileNotFoundError(f"请修改 TRAIN_DIR：{TRAIN_DIR}")
    if not 0 <= OVERLAY_ALPHA <= 1:
        raise ValueError("OVERLAY_ALPHA 必须在 [0,1] 内。")
    if CPU_THREADS > 0:
        torch.set_num_threads(CPU_THREADS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if DEVICE == "auto" else torch.device(DEVICE)
    print("Device:", device, "Torch:", torch.__version__, "Torchvision:", torchvision.__version__)
    model, classes, checkpoint_meta = load_model(device)
    selected = read_selected_rows(classes)
    for row in selected:
        if not (TRAIN_DIR / row["relative_path"]).is_file():
            raise FileNotFoundError(TRAIN_DIR / row["relative_path"])
    reference = None
    if REFERENCE_PREDICTIONS_PATH is not None:
        with REFERENCE_PREDICTIONS_PATH.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not {"relative_path", "predicted_class"}.issubset(reader.fieldnames or []):
                raise ValueError("参考预测需要 relative_path、predicted_class 列。")
            reference = {x["relative_path"]: x["predicted_class"] for x in reader}
        if any(x["relative_path"] not in reference for x in selected):
            raise ValueError("参考预测未覆盖所选验证图片。")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((112, 112), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    summaries, all_maps = [], []
    with torch.inference_mode():
        for index, row in enumerate(selected, start=1):
            path = TRAIN_DIR / row["relative_path"]
            with Image.open(path) as opened:
                image = opened.convert("L").convert("RGB")
            inputs = transform(image).unsqueeze(0).to(device)
            baseline = model(inputs) if VERIFY_FORWARD and index == 1 else None
            with AttentionCapture(model) as capture:
                logits = model(inputs)
                maps, self_mass = capture.arrays()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("分类分数异常。")
            if baseline is not None:
                torch.testing.assert_close(logits, baseline, atol=1e-6, rtol=1e-6)
                print("PASS: 捕获注意力与普通前向结果一致。")
            probabilities = logits.softmax(-1)[0].cpu().numpy()
            prediction = int(probabilities.argmax())
            true_name, predicted_name = classes[row["label"]], classes[prediction]
            folder = OUTPUT_DIR / f"{index:02d}_{true_name}_{path.stem}"
            folder.mkdir(parents=True, exist_ok=True)
            title = f"{row['relative_path']} | True: {true_name} | Pred: {predicted_name} ({probabilities[prediction]:.1%})"
            display = image.resize((112, 112), resample=Image.Resampling.BICUBIC)
            save_figures(display, maps, self_mass, folder, title)
            summary = dict(relative_path=row["relative_path"], true_class=true_name,
                           predicted_class=predicted_name, correct=prediction == row["label"],
                           confidence=float(probabilities[prediction]), output_folder=str(folder))
            if reference is not None:
                summary.update(reference_prediction=reference[row["relative_path"]],
                               matches_reference=predicted_name == reference[row["relative_path"]])
            summary.update({f"prob_{name}": float(p) for name, p in zip(classes, probabilities)})
            (folder / "prediction.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            summaries.append(summary)
            all_maps.append(maps)
            print(f"[{index}/{len(selected)}] {title}", flush=True)
    write_csv(OUTPUT_DIR / "index.csv", summaries)
    mean_maps = np.stack(all_maps).mean(axis=0)
    write_csv(OUTPUT_DIR / "mean_layer_statistics.csv", [
        dict(layer=i + 1, mean_peak=float(np.stack(all_maps)[:, i].max(axis=(1, 2)).mean()),
             peak_of_mean_map=float(mean_maps[i].max())) for i in range(24)
    ])
    notes = dict(checkpoint=str(CHECKPOINT_PATH), validation_csv=str(VAL_SPLIT_PATH), train_dir=str(TRAIN_DIR),
                 checkpoint_metadata=checkpoint_meta, classes=classes, num_images=len(selected),
                 images_per_class=IMAGES_PER_CLASS if not SELECTED_RELATIVE_PATHS else None,
                 selection="explicit" if SELECTED_RELATIVE_PATHS else "first-per-class round robin",
                 formula="s[j] = mean_h mean_i A[h,i,j]; i=query, j=key", includes_self_attention=True,
                 image_size=112, layers=24, heads=16, tokens=196, cls_token=False,
                 color_scale="shared [0,max] over all 24 layers, separately for each image",
                 interpretation="Average received attention, not class attribution or CLS attention.",
                 inference="eval, float32, no gradients, no TTA; eval-only PyTorch reconstruction",
                 torch_version=str(torch.__version__), torchvision_version=str(torchvision.__version__))
    if reference is not None:
        matches = sum(x["matches_reference"] for x in summaries)
        notes.update(reference_matches=matches, reference_total=len(summaries))
        print(f"与导出的验证预测一致：{matches}/{len(summaries)}。")
        if matches != len(summaries):
            print("注意：存在与参考标签不同的预测；请检查版本、输入、权重与浮点精度差异。")
    (OUTPUT_DIR / "visualization_config.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    print("完成：", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
