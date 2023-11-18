from torch.utils.data import Dataset
# import nlpaug.augmenter.word as naw
# from transformers import MistralForCausalLM, MistralForSequenceClassification
import torch
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

collate_fn = partial(collate_factory,IsTrain=True)
collate_inf = partial(collate_factory,IsTrain=False)


#############
### Model ###
#############

class LM(object):
    def __init__(self, model,tokenizer,num_virtual_tokens,alpha,config_type):
        self.model = model
        self.pos_ = tokenizer('Yes yes',add_special_tokens=False,return_attention_mask=False)['input_ids']
        self.neg_ = tokenizer('No no',add_special_tokens=False,return_attention_mask=False)['input_ids']
        self.vocab_size = tokenizer.vocab_size
        self.num_virtual_tokens_ = 0 if (config_type == 'prefix' or config_type == 'LoRA') else num_virtual_tokens 
        self.alpha_ = alpha
        self.loss_bce_ = torch.nn.BCEWithLogitsLoss()
        self.loss_lm_ = torch.nn.CrossEntropyLoss()
        
    def predict(self,input_ids,attention_mask,*args,**kwargs):
        with torch.no_grad():
            out = self.model(input_ids,attention_mask,*args,**kwargs)
            logits = out.logits[:,-1,self.pos_].sum(-1) - out.logits[:,-1,self.neg_].sum(-1)
            return logits
    
    def get_loss(self,input_ids,attention_mask,label,score,*args,**kwargs):
        out = self.model(input_ids,attention_mask,*args,**kwargs)
        loss_bce = self.loss_bce_(out.logits[:,-1,self.pos_].sum(-1) - out.logits[:,-1,self.neg_].sum(-1),label)
        
        # LM objective
        shift_logits = out.logits[..., self.num_virtual_tokens_:-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_lm = self.loss_lm_(shift_logits, shift_labels)
        return loss_bce + self.alpha_ * loss_lm

class Classification(object):
    def __init__(self,model, alpha):
        self.model = model
        self.alpha_ = alpha
        self.loss_bce_ = torch.nn.BCEWithLogitsLoss()
        self.loss_score_ = lambda y,yhat: torch.sum((y!=-1.)*torch.abs(yhat-y))/torch.sum((y!=-1.)+0.01)
        
    def predict(self,input_ids,attention_mask,*args,**kwargs):
        with torch.no_grad():
            out = self.model(input_ids,attention_mask,*args,**kwargs).logits
            return out[:,0]
    
    def get_loss(self,input_ids,attention_mask,label,score,*args,**kwargs):
        out = self.model(input_ids,attention_mask,*args,**kwargs).logits
        return self.loss_bce_(out[:,0], label) + self.alpha_ * self.loss_score_(score,out[:,1])



# config_map = {'prefix':PrefixTuningConfig,
#                 'prompt_encoder':PromptEncoderConfig,
#                 'prompt_txt': PromptTuningConfig,
#                 'LoRA': LoraConfig,}


# def load_config(config_dict):
#     TARGET_MODEL = config_dict["TARGET_MODEL"]
#     pred_type = config_dict["pred_type"]
#     config_type = config_dict["config_type"]
#     epochs = config_dict["epochs"]
#     alpha = config_dict["alpha"]
#     aug_kwargs = config_dict["aug_kwargs"]
#     config_class = config_map[config_dict["config_class"]]
#     config_kwargs = config_dict["config_kwargs"]

#     return TARGET_MODEL, pred_type, config_type, epochs, alpha, aug_kwargs, config_class, config_kwargs
