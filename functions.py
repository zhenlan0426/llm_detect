from torch.utils.data import Dataset, DataLoader
# import nlpaug.augmenter.word as naw
import torch
import torch.nn as nn
from functools import partial

pad_to_multiple_of = 16
max_length = 512

############
### data ###
############
class InfData(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self,idx):
        text = self.df.text[idx]
        return text

class TxtData(InfData):
    def __init__(self, df, aug=None):
        super().__init__(df)
        self.aug = aug

    def __getitem__(self,idx):
        text = self.df.text[idx]
        if self.aug:
            text = self.aug(text)[0]
        label = self.df.label[idx]
        score = self.df.score[idx]
        return text, label, score

def collate_factory(data,tokenizer,IsTrain,prompt,prompt_mask):
    # prompt should be encoded
    if IsTrain:
        text, label, score = zip(*data)
    else:
        text = data
    kwargs = {  'add_special_tokens': True,  # Add '[CLS]' and '[SEP]'
                'max_length': max_length,  # Maximum length of the sequences
                'truncation': True,  # Truncate longer sequences to `max_length`
                'return_attention_mask': True,  # Return attention masks
                'return_tensors': 'pt'  # Return PyTorch tensors
            }
    if prompt is not None:
        kwargs.update({'padding':'max_length','max_length': max_length-prompt.shape[1]})
    else:
        kwargs.update({'pad_to_multiple_of':pad_to_multiple_of,'padding':'longest'})
    text = tokenizer.batch_encode_plus(text,**kwargs)
    input_ids,attention_mask = text['input_ids'], text['attention_mask']
    if prompt is not None:
        n = input_ids.shape[0]
        input_ids = torch.cat([input_ids,prompt.broadcast_to(n,-1)],1)
        attention_mask = torch.cat([attention_mask,prompt_mask.broadcast_to(n,-1)],1)

    if IsTrain:
        return input_ids,attention_mask, torch.tensor(label, dtype=torch.float32), torch.tensor(score, dtype=torch.float32)
    else:
        return input_ids,attention_mask

collate_fn = partial(collate_factory,IsTrain=True,prompt=None,prompt_mask=None)
collate_inf = partial(collate_factory,IsTrain=False,prompt=None,prompt_mask=None)
