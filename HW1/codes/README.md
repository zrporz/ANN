# README

## 代码结构

- `data/`：MINIST数据集
- `log/`：训练日志，运行 `log/draw.py` 中实现了绘制各种图像的函数
- `run_mlp.py`：程序入口，支持通过命令传入参数，如 `python run_mlp --act RELU --loss SoftmaxCrossEntropy --bs 100 --hs 784 --lr 1e-4 --wd 1e-3 --mom 0.9`。和原代码框架相比，`run_mlp.py` 中实现了 `Log` 类，用来记录训练 loss，accuracy，时间等信息
- `layers.py`：实现线性层和各类激活函数，包括 `Relu`，`Sigmoid`，`Selu`，`Swish`，`Gelu`，`Linear` 类，除了修改 #TODO 的内容外，为了实现 dropout 设计（参见report），还调整了 `Linear` 类的初始化方法
- `loss.py`：实现各类损失函数，包括 `MSELoss`，`SoftmaxCrossEntropyLoss`，`HingeLoss`，`FocalLoss`

