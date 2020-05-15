
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import pandas as pd
import numpy as np

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
 
        #tokenize into integer indices
        x = self.src_tokenizer.convert_tokens_to_ids(self.src_tokenizer.tokenize(x))
        y = self.tgt_tokenizer.convert_tokens_to_ids(self.tgt_tokenizer.tokenize(y))

        #add special tokens to target
        y = [self.tgt_tokenizer.bos_token_id] + y + [self.tgt_tokenizer.eos_token_id]

        return torch.LongTensor(x), torch.LongTensor(y)
