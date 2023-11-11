from torch.utils.data import Dataset, DataLoader
import nlpaug.augmenter.word as naw
import torch
import torch.nn as nn


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

class TxtData(Dataset):
    def __init__(self, df, aug=None):
        self.df = df
        self.aug = aug
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self,idx):
        text = self.df.text[idx]
        if self.aug:
            text = self.aug(text)[0]
        label = self.df.label[idx]
        score = self.df.score[idx]
        return text, label, score
    
def collate_fn(data,tokenizer):
    text, label, score = zip(*data)
    text = tokenizer.batch_encode_plus(text,
                                        pad_to_multiple_of=pad_to_multiple_of,
                                        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                        max_length=max_length,  # Maximum length of the sequences
                                        truncation=True,  # Truncate longer sequences to `max_length`
                                        padding=True,  # Pad shorter sequences to `max_length`
                                        return_attention_mask=True,  # Return attention masks
                                        return_tensors='pt'  # Return PyTorch tensors)
    )
    input_ids,attention_mask = text['input_ids'], text['attention_mask']
    return input_ids,attention_mask, torch.tensor(label, dtype=torch.float32), torch.tensor(score, dtype=torch.float32)

def collate_inf(data,tokenizer):
    text = data
    text = tokenizer.batch_encode_plus(text,
                                        pad_to_multiple_of=pad_to_multiple_of,
                                        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                        max_length=max_length,  # Maximum length of the sequences
                                        truncation=True,  # Truncate longer sequences to `max_length`
                                        padding=True,  # Pad shorter sequences to `max_length`
                                        return_attention_mask=True,  # Return attention masks
                                        return_tensors='pt'  # Return PyTorch tensors)
    )
    return text['input_ids'], text['attention_mask']