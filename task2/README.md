# 人脸表情分类注意力可视化

两个脚本均加载已完成表情分类微调的 `best.pt`，无需重新训练。运行前请修改脚本开头的模型权重、验证集清单、图片目录和输出目录路径；默认每类选取 5 张验证图片。

- `visualize_imagenet_vit_attention.py`：用于 **ImageNet 预训练**的 Torchvision ViT-B/16，生成 12 层平均 CLS 注意力图、Attention Rollout 图，以及第 7、8 层各 12 个头的注意力图。
- `visualize_cvlface_received_attention.py`：用于 **人脸数据 WebFace4M 上进行身份识别预训练**的 CVLFace AdaFace ViT，生成 24 层平均接收注意力图。该模型没有 CLS token，注意力先对所有头、再对所有 Query 取平均，保留各 patch 位置的接收权重。

注意力图与输入图片叠加，用于观察读取权重的空间分布，不直接代表某个表情类别的贡献。

依赖：匹配的 PyTorch、Torchvision，以及 NumPy、Pillow、Matplotlib。权重和数据需自行准备，本目录不包含这些文件。

在仓库根目录运行：

```bash
python task2/visualize_imagenet_vit_attention.py
python task2/visualize_cvlface_received_attention.py
```
