"""加载已训练的 ViT-B/16；不训练、不下载预训练权重。

运行：python task2/visualize_imagenet_vit_attention.py
依赖：匹配的 torch/torchvision，以及 numpy、Pillow、matplotlib。
Kaggle 使用时将下面四个路径改成实际 /kaggle/input/... 和 /kaggle/working/... 路径。
输出每张图片的 12 层平均 CLS 注意力、最终 Rollout、Layer 7/8 逐头图和原始数值。
"""

from pathlib import Path

# ==================== 文件路径：优先修改这里 ====================
TASK_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = TASK_DIR / "result" / "best.pt"
VAL_SPLIT_PATH = TASK_DIR / "result" / "val_split.csv"
# CSV 中的原始 path 是 Kaggle 路径；脚本使用 relative_path 在本地重新定位图片。
TRAIN_DIR = TASK_DIR / "dataset" / "2026task2" / "fer_data" / "fer_data" / "train"
OUTPUT_DIR = TASK_DIR / "result" / "attention_visualizations"

# ==================== 可视化配置 ====================
IMAGES_PER_CLASS = 5  # 从验证清单中每类取前 5 张，共 30 张，不根据结果挑选。
# 留空则自动选图；填写后按指定顺序处理，且必须属于验证集。
SELECTED_RELATIVE_PATHS = []  # 例：["Neutral/dpnhvazzvq.jpg", "Fear/xxx.jpg"]
DEVICE = "auto"  # "auto"、"cpu" 或 "cuda"
CPU_THREADS = 4
OVERLAY_ALPHA = 0.45
COLORMAP = "inferno"
DPI = 160
SAVE_ATTENTION_ARRAYS = True
VERIFY_FORWARD = True  # 第一张图对比普通前向，确认获取注意力没有实质改变预测。
HEAD_LAYERS = (7, 8)  # 从 1 开始编号；每层展开 12 个头。
APPEND_ONLY = True  # 已存在平均图/Rollout 时保留它们，仅追加逐头图。
SAVE_COMBINED_FIGURE = True  # 纵向拼接：12层平均图 -> Layer 7逐头 -> Layer 8逐头。

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
from torchvision.models import vit_b_16
from torchvision.transforms import InterpolationMode


class EmotionModel(nn.Module):
    """与 train_imagenet_vit_fer.py 完全相同的推理结构。"""
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = vit_b_16(weights=None)  # 全部参数随后由 best.pt 严格覆盖。
        self.backbone.heads = nn.Identity()
        self.head = nn.Sequential(nn.LayerNorm(768), nn.Dropout(0.3), nn.Linear(768, num_classes))

    def forward(self, x):
        return self.head(self.backbone(x))


def load_model(device):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    expected = "torchvision/vit_b_16/IMAGENET1K_V1"
    if checkpoint.get("config", {}).get("MODEL_ID") != expected:
        raise ValueError("该脚本仅支持 train_imagenet_vit_fer.py 的 best.pt，不支持 CVLFace。")
    classes = checkpoint["class_names"]
    if len(classes) != len(set(classes)) or not all(isinstance(x, str) for x in classes):
        raise ValueError("权重中的类别名称无效。")
    if checkpoint["config"].get("IMAGE_SIZE") != 224:
        raise ValueError("该权重的输入尺寸不是 224，无法使用当前结构。")
    model = EmotionModel(len(classes))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    model.requires_grad_(False)
    metadata = {k: checkpoint.get(k) for k in ["stage", "epoch", "best_macro_f1", "val_accuracy"]}
    return model, classes, metadata


def read_validation_rows(classes):
    with VAL_SPLIT_PATH.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not {"relative_path", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("验证清单需要 relative_path 和 label 列。")
        rows = list(reader)
    seen = set()
    for row in rows:
        relative = row["relative_path"].replace("\\", "/")
        parts = PurePosixPath(relative)
        if parts.is_absolute() or ":" in relative or ".." in parts.parts or len(parts.parts) < 2:
            raise ValueError(f"非法相对路径：{relative}")
        if relative in seen:
            raise ValueError(f"验证清单包含重复图片：{relative}")
        seen.add(relative)
        label = int(row["label"])
        if not 0 <= label < len(classes) or parts.parts[0] != classes[label]:
            raise ValueError(f"图片路径与权重类别顺序不一致：{relative}")
        if row.get("class_name") and row["class_name"] != classes[label]:
            raise ValueError(f"class_name 与 label 不一致：{relative}")
        row.update(relative_path=relative, label=label)
    if not rows:
        raise ValueError("验证清单为空。")
    return rows


def select_rows(rows, classes):
    if SELECTED_RELATIVE_PATHS:
        keys = [x.replace("\\", "/") for x in SELECTED_RELATIVE_PATHS]
        if len(keys) != len(set(keys)):
            raise ValueError("指定图片中有重复路径。")
        by_path = {x["relative_path"]: x for x in rows}
        missing = [x for x in keys if x not in by_path]
        if missing:
            raise ValueError(f"指定图片不属于当前验证集：{missing}")
        return [by_path[x] for x in keys]
    if IMAGES_PER_CLASS < 1:
        raise ValueError("IMAGES_PER_CLASS 必须为正整数。")
    # 轮询各类别，严格每类相同数量，不按预测结果筛选。
    groups = [[x for x in rows if x["label"] == label] for label in range(len(classes))]
    insufficient = [classes[i] for i, group in enumerate(groups) if len(group) < IMAGES_PER_CLASS]
    if insufficient:
        raise ValueError(f"以下类别不足 {IMAGES_PER_CLASS} 张验证图片：{insufficient}")
    selected = []
    for index in range(IMAGES_PER_CLASS):
        for group in groups:
            selected.append(group[index])
    return selected


class AttentionCapture:
    """临时要求 MHA 返回逐头权重；退出时移除所有 hook。"""
    def __init__(self, model):
        self.layers = list(model.backbone.encoder.layers)
        if len(self.layers) != 12:
            raise ValueError(f"预期 12 个 EncoderBlock，实际 {len(self.layers)}。")
        self.handles = []
        self.weights = [None] * len(self.layers)
        self.head_cls_weights = {}

    @staticmethod
    def request_weights(module, args, kwargs):
        kwargs = dict(kwargs)
        kwargs.update(need_weights=True, average_attn_weights=False)
        return args, kwargs

    def save_weights(self, index):
        def hook(module, inputs, output):
            weights = output[1]
            if weights is None or tuple(weights.shape) != (1, 12, 197, 197):
                raise RuntimeError("没有得到 [1,12,197,197] 逐头注意力权重。")
            # head mean；保留完整 token-to-token 矩阵，供 Rollout 使用。
            self.weights[index] = weights[0].detach().float().mean(dim=0).cpu()
            if index + 1 in HEAD_LAYERS:
                self.head_cls_weights[index + 1] = weights[0, :, 0, :].detach().float().cpu()
        return hook

    def __enter__(self):
        for i, layer in enumerate(self.layers):
            attention = layer.self_attention
            self.handles.append(attention.register_forward_pre_hook(self.request_weights, with_kwargs=True))
            self.handles.append(attention.register_forward_hook(self.save_weights(i)))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()

    def stack(self):
        if any(x is None for x in self.weights):
            raise RuntimeError("部分层没有捕获到注意力。")
        attention = torch.stack(self.weights)
        if not torch.isfinite(attention).all() or (attention < 0).any():
            raise FloatingPointError("注意力包含非有限值或负值。")
        torch.testing.assert_close(attention.sum(-1), torch.ones(12, 197), atol=1e-4, rtol=1e-4)
        return attention

    def head_maps(self):
        result = {}
        for layer in HEAD_LAYERS:
            weights = self.head_cls_weights[layer]
            if not torch.isfinite(weights).all() or (weights < 0).any():
                raise FloatingPointError(f"Layer {layer} 逐头注意力数值异常。")
            torch.testing.assert_close(weights.sum(-1), torch.ones(12), atol=1e-4, rtol=1e-4)
            maps = weights[:, 1:].reshape(12, 14, 14)
            # 确认各头求平均，确实还原原来的平均 CLS 图。
            expected = self.weights[layer - 1][0, 1:].reshape(14, 14)
            torch.testing.assert_close(maps.mean(0), expected, atol=1e-7, rtol=1e-5)
            result[layer] = maps.numpy()
        return result


def attention_maps(attention):
    """输入 [L,N,N]；输出原始 CLS 图及逐层累计 Rollout 图，均为 [L,14,14]。"""
    layers, tokens, other_tokens = attention.shape
    if tokens != 197 or other_tokens != tokens:
        raise ValueError("需要包含 CLS 的 197x197 注意力矩阵。")
    identity = torch.eye(tokens, dtype=attention.dtype)
    joint = identity.clone()
    rollouts = []
    for matrix in attention:
        # A+I：以相同权重近似注意力分支与残差分支，再按行归一化。
        augmented = matrix + identity
        augmented = augmented / augmented.sum(-1, keepdim=True)
        # 当前层在左，累计输入至当前层的信息传播。
        joint = augmented @ joint
        rollouts.append(joint[0, 1:].reshape(14, 14))
    return attention[:, 0, 1:].reshape(layers, 14, 14).numpy(), torch.stack(rollouts).numpy()


def upsample_grid(grid):
    tensor = torch.from_numpy(np.asarray(grid, dtype=np.float32).copy())[None, None]
    return F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()


def overlay(ax, image, heatmap, title, vmax):
    ax.imshow(image)
    ax.imshow(upsample_grid(heatmap), cmap=COLORMAP, vmin=0, vmax=vmax, alpha=OVERLAY_ALPHA)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def save_figures(image, cls_maps, rollout_maps, folder, title):
    # 原始 CLS 图共用色标，支持同一图片跨层比较绝对权重；不逐图 min-max。
    cls_vmax = max(float(cls_maps.max()), 1e-12)
    fig, axes = plt.subplots(3, 4, figsize=(13, 10), layout="constrained")
    for i, ax in enumerate(axes.flat):
        overlay(ax, image, cls_maps[i], f"Layer {i + 1}", cls_vmax)
    fig.suptitle(title + "\nMean CLS-to-patch attention | shared scale across 12 layers", fontsize=12)
    scalar = plt.cm.ScalarMappable(norm=Normalize(0, cls_vmax), cmap=COLORMAP)
    fig.colorbar(scalar, ax=list(axes.flat), shrink=0.7, label="Attention weight (12-head mean)")
    fig.savefig(folder / "cls_attention_12_layers.png", dpi=DPI)
    plt.close(fig)

    final_rollout = rollout_maps[-1]
    rollout_vmax = max(float(final_rollout.max()), 1e-12)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    axes[0].imshow(image)
    axes[0].set_title("Input image")
    axes[0].axis("off")
    overlay(axes[1], image, cls_maps[-1], "Layer 12: mean CLS attention", cls_vmax)
    overlay(axes[2], image, final_rollout, "Attention Rollout: layers 1-12", rollout_vmax)
    fig.suptitle(title, fontsize=11)
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, cls_vmax), cmap=COLORMAP),
                 ax=axes[1], shrink=0.75, label="CLS attention")
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, rollout_vmax), cmap=COLORMAP),
                 ax=axes[2], shrink=0.75, label="Rollout weight")
    fig.savefig(folder / "attention_rollout.png", dpi=DPI)
    plt.close(fig)
    image.save(folder / "input_224.png")
    if SAVE_ATTENTION_ARRAYS:
        np.savez_compressed(folder / "attention_maps.npz", cls_attention=cls_maps,
                            cumulative_rollout=rollout_maps)


def save_head_figures(image, cls_maps, head_maps, folder, title):
    # 两层的 24 个头共用同一个色标，不对每个头单独拉伸亮度。
    head_vmax = max(max(float(x.max()) for x in head_maps.values()), 1e-12)
    for layer, maps in head_maps.items():
        fig, axes = plt.subplots(3, 4, figsize=(13, 10), layout="constrained")
        for h, ax in enumerate(axes.flat):
            overlay(ax, image, maps[h], f"Layer {layer} | Head {h + 1}", head_vmax)
        fig.suptitle(title + f"\nLayer {layer}: 12 separate CLS attention heads | shared scale for layers 7 & 8",
                     fontsize=11)
        scalar = plt.cm.ScalarMappable(norm=Normalize(0, head_vmax), cmap=COLORMAP)
        fig.colorbar(scalar, ax=list(axes.flat), shrink=0.7, label="Per-head CLS-to-patch weight")
        fig.savefig(folder / f"layer{layer}_cls_attention_12_heads.png", dpi=DPI)
        plt.close(fig)
    if SAVE_ATTENTION_ARRAYS:
        np.savez_compressed(folder / "head_attention_maps.npz",
                            **{f"layer{layer}_cls_heads": maps for layer, maps in head_maps.items()})
    if SAVE_COMBINED_FIGURE:
        sections = [("Mean", cls_maps, max(float(cls_maps.max()), 1e-12))]
        sections.extend((f"Layer {layer}", maps, head_vmax) for layer, maps in head_maps.items())
        fig, axes = plt.subplots(3 * len(sections), 4, figsize=(13, 10 * len(sections)),
                                 layout="constrained")
        for section, (name, maps, vmax) in enumerate(sections):
            section_axes = axes[3 * section:3 * section + 3].flatten()
            for i, ax in enumerate(section_axes):
                subtitle = f"Layer {i + 1} | Mean of 12 heads" if section == 0 else f"{name} | Head {i + 1}"
                overlay(ax, image, maps[i], subtitle, vmax)
            label = "12-head mean weight" if section == 0 else "Per-head weight (shared L7/L8 scale)"
            fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, vmax), cmap=COLORMAP),
                         ax=list(section_axes), shrink=0.75, label=label)
        fig.suptitle(title + "\nTop: 12 layer means | Middle: Layer 7 heads | Bottom: Layer 8 heads", fontsize=11)
        fig.savefig(folder / "attention_layers_then_heads.png", dpi=DPI)
        plt.close(fig)


def main():
    for path in [CHECKPOINT_PATH, VAL_SPLIT_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f"请修改文件顶部的路径：{path}")
    if not TRAIN_DIR.is_dir():
        raise FileNotFoundError(f"请将 TRAIN_DIR 指向类别文件夹的父目录：{TRAIN_DIR}")
    if not 0 <= OVERLAY_ALPHA <= 1:
        raise ValueError("OVERLAY_ALPHA 必须在 [0,1] 内。")
    if tuple(HEAD_LAYERS) != (7, 8):
        raise ValueError("当前追加布局要求 HEAD_LAYERS=(7,8)。")
    if CPU_THREADS > 0:
        torch.set_num_threads(CPU_THREADS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if DEVICE == "auto" else torch.device(DEVICE)
    print("Device:", device, "Torch:", torch.__version__, "Torchvision:", torchvision.__version__)
    model, classes, checkpoint_meta = load_model(device)
    selected = select_rows(read_validation_rows(classes), classes)
    paths = [TRAIN_DIR / x["relative_path"] for x in selected]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"找不到验证图片，请检查 TRAIN_DIR：{missing[:5]}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    summaries = []
    with torch.inference_mode():
        for index, (row, path) in enumerate(zip(selected, paths), start=1):
            with Image.open(path) as opened:
                original = opened.convert("L").convert("RGB")
            inputs = transform(original).unsqueeze(0).to(device)
            baseline = model(inputs) if VERIFY_FORWARD and index == 1 else None
            with AttentionCapture(model) as capture:
                logits = model(inputs)
                attention = capture.stack()
                head_maps = capture.head_maps()
            if baseline is not None:
                torch.testing.assert_close(logits, baseline, atol=2e-4, rtol=2e-4)
                print("PASS: 获取注意力的前向计算与普通前向结果一致（浮点容差内）。")
            probabilities = logits.softmax(-1)[0].cpu().numpy()
            prediction = int(probabilities.argmax())
            cls_maps, rollouts = attention_maps(attention)
            true_name, predicted_name = classes[row["label"]], classes[prediction]
            folder = OUTPUT_DIR / f"{index:02d}_{true_name}_{path.stem}"
            folder.mkdir(parents=True, exist_ok=True)
            title = f"{row['relative_path']} | True: {true_name} | Pred: {predicted_name} ({probabilities[prediction]:.1%})"
            display = original.resize((224, 224), resample=Image.Resampling.BILINEAR)
            existing = [folder / "cls_attention_12_layers.png", folder / "attention_rollout.png"]
            if not APPEND_ONLY or not all(x.is_file() for x in existing):
                save_figures(display, cls_maps, rollouts, folder, title)
            save_head_figures(display, cls_maps, head_maps, folder, title)
            summary = dict(relative_path=row["relative_path"], true_class=true_name,
                           predicted_class=predicted_name, correct=prediction == row["label"],
                           confidence=float(probabilities[prediction]), output_folder=str(folder))
            summary.update({f"prob_{name}": float(p) for name, p in zip(classes, probabilities)})
            (folder / "prediction.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            summaries.append(summary)
            print(f"[{index}/{len(selected)}] {title}", flush=True)
    with (OUTPUT_DIR / "index.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    notes = dict(checkpoint=str(CHECKPOINT_PATH), validation_csv=str(VAL_SPLIT_PATH),
                 train_dir=str(TRAIN_DIR), classes=classes, checkpoint_metadata=checkpoint_meta,
                 device=str(device), torch_version=str(torch.__version__),
                 torchvision_version=str(torchvision.__version__), num_images=len(selected),
                 selection="explicit" if SELECTED_RELATIVE_PATHS else "first-per-class round robin",
                 images_per_class=None if SELECTED_RELATIVE_PATHS else IMAGES_PER_CLASS,
                 head_fusion="arithmetic mean over 12 heads", image_size=224,
                 expanded_head_layers=list(HEAD_LAYERS), append_only=APPEND_ONLY,
                 expanded_head_scale="shared [0, max] over both layers' 24 heads, within each image",
                 combined_figure=SAVE_COMBINED_FIGURE,
                 raw_attention_scale="shared [0, max] within each image's 12 layers",
                 rollout="A=head_mean; R=(A+I)/row_sum(A+I); joint=R_layer @ joint_previous",
                 rollout_caveat="Approximation: ignores MLP, value projections and actual residual magnitudes.",
                 interpretation="Attention weights are not class-specific attribution or causal importance.",
                 inference="eval, float32, no augmentation, no TTA, no gradients")
    (OUTPUT_DIR / "visualization_config.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    print("完成：", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
