# 英文文本关系抽取
### 基于 pytorch、transformers、XLNet
数据集来自[这里](https://god.yanxishe.com/82)

解决方案： Huggingface 的 transformer 库中的预训练模型 XLNet 做 tokenize 和分类任务。

将所有可能的分类看作不同的类别，如因果关系的 Cause-Effect(e2,e1) 和 Cause-Effect(e1,e2)，共18分类。

`data.py` 处理原始数据，用 tokenizer 对训练数据进行编码并返回 dataloader

`model.py` 模型的保存加载、预测和验证等

`main.py` 训练过程和推理过程

[参考](https://wmathor.com/index.php/archives/1483/)
