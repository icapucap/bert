
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertModel, BertForMaskedLM, BertTokenizer

from pathlib import Path

def split_data(file_path, destination):

    data = pd.read_csv(file_path, sep='\t',error_bad_lines=False, header=None, engine='python',quotechar='"')[[0, 1]]

    mask = np.random.rand(data.shape[0]) < 0.8
    train = data[mask]
    valid = data[~mask]
    # print(valid.shape)

    train.to_csv(Path(destination) / 'train.csv', header=False, index=False)

    mask_test = np.random.rand(valid.shape[0]) < 0.5
    test = valid[mask_test]
    valid = valid[~mask_test]
    # print(test.shape,valid.shape)


    valid.to_csv(Path(destination) / 'valid.csv', header=False, index=False)
    test.to_csv(Path(destination) / 'test.csv', header=False, index=False)
# split_data('/home/shidhu/itr/itr/hin.txt', '/home/shidhu/itr/itr/')


class PadSequence:
    
    def __init__(self, src_padding_value, tgt_padding_value):
        self.src_padding_value = src_padding_value
        self.tgt_padding_value = tgt_padding_value
    
    def __call__(self, batch):
        
        x = [s[0] for s in batch]
        x = pad_sequence(x, 
                         batch_first=True, 
                         padding_value=self.src_padding_value)

        y = [s[1] for s in batch]
        y = pad_sequence(y, 
                         batch_first=True, 
                         padding_value=self.tgt_padding_value)

        return x, y


class IndicDataset(Dataset):
  
    def __init__(self, 
                 src_tokenizer,
                 tgt_tokenizer,
                 destination,
                 is_train=True,is_test=False):
        if is_test:
            destination += 'test.csv'
        else:
            destination += 'train.csv' if is_train else 'valid.csv'
        self.df = pd.read_csv(destination, header=None, engine='python',quotechar='"')

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        y, x = self.df.loc[index]
        # print(y)    #english
        # print(x)    #hindi
 
        #tokenize into integer indices
        x = self.src_tokenizer.convert_tokens_to_ids(self.src_tokenizer.tokenize(x))
        y = self.tgt_tokenizer.convert_tokens_to_ids(self.tgt_tokenizer.tokenize(y))

        #add special tokens to target
        y = [self.tgt_tokenizer.bos_token_id] + y + [self.tgt_tokenizer.eos_token_id]

        return torch.LongTensor(x), torch.LongTensor(y)


# src_tokenizers = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# tgt_tokenizers = BertTokenizer.from_pretrained('bert-base-uncased')

# tgt_tokenizers.bos_token = '<s>'
# tgt_tokenizers.eos_token = '</s>'

# def prepare_data():
#         from data import split_data
#         split_data('/home/shidhu/itr/itr/hin.txt', '/home/shidhu/itr/itr/')

    
# def train_dataloader():
#     from data import IndicDataset, PadSequence
#     pad_sequence = PadSequence(src_tokenizers.pad_token_id, tgt_tokenizers.pad_token_id)

#     return DataLoader(IndicDataset(src_tokenizers, tgt_tokenizers, '/home/shidhu/itr/itr/', True, False), 
#                             batch_size=64, 
#                             shuffle=False, 
#                             collate_fn=pad_sequence)

# prepare_data()
# dataloader = train_dataloader()

# for x,y in enumerate(dataloader):
#     print('src')
#     print(x)
#     print('target')
#     print(y)
#     print()
#     print(y[0])
#     print()
#     print(y.shape)
#     print('*'*20)

# print(len(dataloader))
# x = src_tokenizers.convert_tokens_to_ids(src_tokenizers.tokenize('Get out!'))
# y = tgt_tokenizers.convert_tokens_to_ids(tgt_tokenizers.tokenize('बाहर निकल जाओ!'))
# print(x)
# print(y)
# y = [tgt_tokenizers.bos_token_id] + y + [tgt_tokenizers.eos_token_id]
# print()
# print(y)
# print()
# print()
# print(torch.LongTensor(x))
# print(torch.LongTensor(y))

# # [14439, 14942, 106]       Go away!
# # [1318, 29870, 100, 999]   चले जाओ!

# # [100, 1318, 29870, 100, 999, 100]


# # tensor([14439, 14942,   106])
# # tensor([  100,  1318, 29870,   100,   999,   100])

# dataloader = DataLoader([torch.LongTensor(x),torch.LongTensor(y)],batch_size=64,shuffle=False, 
#                             collate_fn=pad_sequence)
# print(dataloader)
# for a,b in enumerate(dataloader):
#     # print(a)
#     print(b)
#     print(b[0])
#     print(type(b[0]),b[0].shape)
#     # print(b[1].shape)