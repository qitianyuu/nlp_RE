"""
# File       :  model.py
# Time       :  2022/1/13 8:43 下午
# Author     : Qi
# Description:
"""
import csv

import numpy as np
import torch
import os
import torch.utils.data as Data
from transformers import XLNetForSequenceClassification
from data import convert
import pandas as pd

output_dir = './models/'
output_model_file = os.path.join(output_dir, 'pytorch_model.bin')
output_config_file = os.path.join(output_dir, 'config.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten() # [3, 5, 8, 1, 2, ....]
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def eval(model, validation_dataloader):
    model.eval()
    eval_loss, eval_accuracy, nb_eval_steps = 0, 0, 0
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2])[0]
            logits = logits.detach().cpu().numpy()
            label_ids = batch[3].cpu().numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
    global best_score
    if best_score < eval_accuracy / nb_eval_steps:
        best_score = eval_accuracy / nb_eval_steps
        save(model)

def save(model):
    torch.save(model.state_dict(), output_model_file)
    model.config.to_json_file(output_config_file)

def predict(class_list):
    checkpoint = 'xlnet-base-cased'
    model = XLNetForSequenceClassification.from_pretrained(output_dir).to(device)

    with open('test.csv') as f:
        rows = [row for row in csv.reader(f)]
        rows = np.array(rows[1:])
        sentences = [text for idx, text in rows]

    input_ids, token_type_ids, attention_mask, labels = convert(checkpoint, sentences, [0])
    dataset = Data.TensorDataset(input_ids, token_type_ids, attention_mask)
    dataloader = Data.DataLoader(dataset, 32, False)

    model.eval()
    predict = []
    for idx, batch in enumerate(dataloader):
        print('{} batch start..'.format(idx))
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            log = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2])[0]
            log = log.detach().cpu().numpy()
            pred = np.argmax(log, axis=1).flatten()
            predict.extend(pred)
        print('{} batch finish..'.format(idx))

    predict = [class_list[p] for p in predict]

    pd.DataFrame(data=predict, index=range(len(predict))).to_csv('pred.csv')




