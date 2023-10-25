# README

本目录下实现了cnn网络，模型定义在 `model.py` 中，模型参数可以参考 `/config` 下设置，可以通过运行 `python main.py --config_path "./config/config7.json" --num_epochs 100 --is_train` 进行训练，训练日志记录在 `/log` 下，可以通过 tensorboard 查看

可以通过调整 `model.py` 改变模型结构顺序