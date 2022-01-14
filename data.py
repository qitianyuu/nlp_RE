"""
# File       :  data.py
# Time       :  2022/1/10 3:51 下午
# Author     : Qi
# Description:
"""
import csv
import os
import numpy as np
import torch
import torch.utils.data as Data
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

def data_process(path):
    """
    数据预处理
    :param path: 路径
    :return: 句子， 标签， 标签类， 标签数量
    """
    assert os.path.exists(path), "No such file or dictionary! Double-check your file path."

    with open(path) as f:
        rows = [row for row in csv.reader(f)]
        rows = np.array(rows[1:])
        sentences, labels = [sentence for sentence, _ in rows], [label for _, label in rows]
        classes = list(set(labels))
        num_classes = len(classes)
        labels = [classes.index(label) for label in labels]
    return sentences, labels, classes, num_classes

def convert(checkpoint, sentences, labels):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    input_ids, token_type_ids, attention_mask = [], [], []
    for i in range(len(sentences)):
        encoded_dict = tokenizer.encode_plus(
            sentences[i],  # 输入文本
            add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
            max_length=96,  # 不够填充
            pad_to_max_length=True,
            return_tensors='pt',  # 返回 pytorch tensors 格式的数据
        )
        # print(encoded_dict['input_ids'].shape)
        input_ids.append(encoded_dict['input_ids'])
        token_type_ids.append(encoded_dict['token_type_ids'])
        attention_mask.append(encoded_dict['attention_mask'])


    input_ids = torch.LongTensor(torch.cat(input_ids, dim=0))
    token_type_ids = torch.LongTensor(torch.cat(token_type_ids, dim=0))
    attention_mask = torch.LongTensor(torch.cat(attention_mask, dim=0))
    labels = torch.tensor(labels)

    return input_ids, token_type_ids, attention_mask, labels


def train_test_spl(input_ids, token_type_ids, attention_mask, labels, batch_size):
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, random_state=1, test_size=0.1)
    train_token, val_token, _, _ = train_test_split(token_type_ids, labels, random_state=1, test_size=0.1)
    train_mask, val_mask, _, _ = train_test_split(attention_mask, labels, random_state=1, test_size=0.1)

    train_data = Data.TensorDataset(train_inputs, train_token, train_mask, train_labels)
    train_dataloader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    validation_data = Data.TensorDataset(val_inputs, val_token, val_mask, val_labels)
    validation_dataloader = Data.DataLoader(validation_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, validation_dataloader




