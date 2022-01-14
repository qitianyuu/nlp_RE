"""
# File       :  main.py
# Time       :  2022/1/10 3:42 下午
# Author     : Qi
# Description:
"""
from data import data_process, convert, train_test_spl
from transformers import XLNetForSequenceClassification
from transformers import AdamW, AutoTokenizer
from model import eval, predict
import torch


batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = 'xlnet-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
sentences, labels, classes, num_classes = data_process('train.csv')
input_ids, token_type_ids, attention_mask, labels = convert(checkpoint, sentences, labels)
train_dataloader, validation_dataloader = train_test_spl(input_ids, token_type_ids, attention_mask, labels, batch_size)


model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=num_classes).to(device)


param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    'weight_decay_rate': 0.0}]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

for _ in range(2):
    for i, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        loss = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2], labels=batch[3])[0]
        print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            eval(model, validation_dataloader)

predict(classes)



