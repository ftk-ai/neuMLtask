# 中心凹检测与定位实验

`neu-mltask3-4.ipynb` 是在 Kaggle 上训练和运行的 Jupyter Notebook 文件。

模型采用 **YOLO11 + MobileNetV3-Small** 的两阶段结构：先由 YOLO11n 检测并粗定位中心凹所在区域，再由 MobileNetV3-Small 精修中心凹坐标。