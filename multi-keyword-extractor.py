from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
import torch
import argparse
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='BERT Keyword Extractor')
parser.add_argument('--sentence', type=str, default=' ',
                    help='sentence to get keywords')
parser.add_argument('--path', type=str, default='model.pt',
                    help='path to load model')
args = parser.parse_args()

# tag2idx = {'B': 0, 'I': 1, 'O': 2}
# tags_vals = ['B', 'I', 'O']
tag2idx = {'B-0': 0, 'B-1': 1, 'B-2': 2, 
          'B-3': 3, 'B-4': 4, 'O': 5}
tags_vals = ['B-1','B-0','B-2',
          'B-3', 'B-4','O']

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx))

def keywordextract(sentence, path):
    text = sentence
    tkns = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tkns)
    segments_ids = [0] * len(tkns)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([segments_ids]).to(device)
    model = torch.load(path)
    model.eval()
    prediction = []
    logit = model(tokens_tensor, token_type_ids=None,
                                  attention_mask=segments_tensors)
    logit = logit.detach().cpu().numpy()
    prediction.extend([list(p) for p in np.argmax(logit, axis=2)])
    print(prediction)
    for k, j in enumerate(prediction[0]):
        if j!=5:
            print(tokenizer.convert_ids_to_tokens(tokens_tensor[0].to('cpu').numpy())[k], j)
            

keywordextract(args.sentence, args.path)
