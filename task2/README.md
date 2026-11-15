# 人脸表情分类训练

- `train_imagenet_vit_fer.py`：使用 **ImageNet 预训练**的 Torchvision ViT-B/16，微调用于六类人脸表情分类，输入大小为 224×224。
- `train_cvlface_vit_fer.py`：使用 **人脸数据 WebFace4M 上进行身份识别预训练**的 CVLFace AdaFace ViT，微调用于相同任务，输入大小为 112×112。

两个脚本均适用于 Kaggle。运行前修改文件开头的数据和输出路径，并按脚本说明安装依赖。训练先冻结骨干训练分类头，再解冻全模型微调，保存最佳权重并生成测试集提交文件。
