# README
利用 transformer 模型完成句子生成任务
## 训练和测试
```python
python main.py --name ${task_name} # 从随机初始化开始训练
python main.py --name ft_1-2-3 --pretrain_dir ./pretrained/L-3 # 从预训练模型开始训练
python main.py --test ${task_name} --decode_strategy random --temperature 1.0 # 测试, 可以选择不同的解码策略和超参数, 可以通过查看代码来调整
```
