import os
import argparse
from nltk.tokenize import sent_tokenize
import abbreviation
from abbreviation import limits


parser = argparse.ArgumentParser(description='BERT Keyword Extraction Model')

parser.add_argument('--data', type=str, default='maui-semeval2010-train',
                    help='location of the data corpus')
parser.add_argument('--epochs', type=int, default=4,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--seq_len', type=int, default=75, metavar='N',
                    help='sequence length')
parser.add_argument('--lr', type=float, default=3e-5,
                    help='initial learning rate')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

# input as file path, output as sentences and corresponding BIO tag representations
# Data preprocessing
def convert(path):
  with open("/content/Partitive-Files/" + path, "r") as f: 
    lines = f.readlines()
  lines = [line.split() for line in lines]
  sentences = []
  tags = []
  sentence = []
  tag = []
  for line in lines:
    if len(line) != 0:
      sentence.append(line[0])
      if len(line) >= 6:
        tag_str = ""
        if line[5].startswith("ARG"):
          if line[5] == "ARG1":
            tag.append("B-1")
          elif line[5] == "ARG0":
            tag.append("B-0")
          elif line[5] == "ARG2":
            tag.append("B-2")
          elif line[5] == "ARG3":
            tag.append("B-3")
          elif line[5] == "ARG3":
            tag.append("B-4")
        #   if len(tag) == 0:
        #     tag_str = "B"
        #   else:
        #     if tag[-1] == "B":
        #       tag.append("I")
        #     else:
        #       tag.append("B")
        # else:
        #   tag.append("O")
      else:
        tag.append("O")
    else:
      sentences.append(" ".join(sentence))
      tags.append(tag)
      sentence = []
      tag = []
      continue
  if len(sentence) != 0:
    sentences.append(" ".join(sentence))
  if len(tag) != 0:
    tags.append(tag)
  return sentences, tags

train_path = args.data
partitives = [f for f in os.listdir(train_path) if f.startswith("partitive")]
percents = [f for f in os.listdir(train_path) if f.startswith("%")]

train_sents = []
train_tags = []
test_sents = []
test_tags = []
# print(partitives)
for path in partitives: 
  print(path)
  if path.endswith(".train"):
    train_sents, train_tags = convert(path)
    print(len(train_sents))
  if path.endswith(".test"):
    test_sents, test_tags = convert(path)
# print(train_sents[0])
# print(len(train_tags))
# print(len(test_sents))
# print(len(test_tags))



# ## Model

import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam

MAX_LEN = args.seq_len
bs = args.batch_size


# tag2idx = {'B': 0, 'I': 1, 'O': 2}
# tags_vals = ['B', 'I', 'O']
# tag2idx = {'B-1': 0, 'I-1': 1, 'B-0': 2, 'I-0': 3, 'B-2': 4, 'I-2': 5, 
#           'B-3': 6, 'I-3': 7, 'B-4': 9, 'I-4': 11, 'O': 12}
# tags_vals = ['B-1','I-1','B-0','I-0','B-2','I-2',
#           'B-3', 'I-3', 'B-4', 'I-4','O']
tag2idx = {'B-0': 0, 'B-1': 1, 'B-2': 2, 
          'B-3': 3, 'B-4': 4, 'O': 5}
tags_vals = ['B-1','B-0','B-2',
          'B-3', 'B-4','O']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenized_train = [tokenizer.tokenize(sent) for sent in train_sents]
tokenized_test = [tokenizer.tokenize(sent) for sent in test_sents]

input_ids_train = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_train],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
input_ids_test = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_test],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

tr_tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in train_tags],
                     maxlen=MAX_LEN, value=tag2idx["O"], padding="post",
                     dtype="long", truncating="post")
val_tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in test_tags],
                     maxlen=MAX_LEN, value=tag2idx["O"], padding="post",
                     dtype="long", truncating="post")

tr_attention_masks = [[float(i>0) for i in ii] for ii in input_ids_train]
val_attention_masks = [[float(i>0) for i in ii] for ii in input_ids_test]

tr_inputs = torch.tensor(input_ids_train)
val_inputs = torch.tensor(input_ids_test)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_attention_masks)
val_masks = torch.tensor(val_attention_masks)

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)


model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx))


model = model.cuda()


FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters()) 
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=args.lr)


from seqeval.metrics import f1_score

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


epochs = args.epochs
max_grad_norm = 1.0

for _ in trange(epochs, desc="Epoch"):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # forward pass
        loss = model(b_input_ids, token_type_ids=None,
                     attention_mask=b_input_mask, labels=b_labels)
        # backward pass
        loss.backward()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        model.zero_grad()
    # print train loss per epoch
    print("Train loss: {}".format(tr_loss/nb_tr_steps))
    # VALIDATION on validation set
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        
        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss/nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    pred_tags = [[tags_vals[p_i] for p in predictions for p_i in p]]
    valid_tags = [[tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]]
    print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))

torch.save(model, args.save)
    
model.eval()
predictions = []
true_labels = []
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
for batch in valid_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                              attention_mask=b_input_mask, labels=b_labels)
        logits = model(b_input_ids, token_type_ids=None,
                       attention_mask=b_input_mask)
        
    logits = logits.detach().cpu().numpy()
    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

    label_ids = b_labels.to('cpu').numpy()
    true_labels.append(label_ids)
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)

    eval_loss += tmp_eval_loss.mean().item()
    eval_accuracy += tmp_eval_accuracy

    nb_eval_examples += b_input_ids.size(0)
    nb_eval_steps += 1

pred_tags = [[tags_vals[p_i] for p_i in p] for p in predictions]
valid_tags = [[tags_vals[l_ii] for l_ii in l_i] for l in true_labels for l_i in l ]
print("Validation loss: {}".format(eval_loss/nb_eval_steps))
print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))  


