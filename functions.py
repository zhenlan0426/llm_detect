from torch.utils.data import Dataset, DataLoader
# import nlpaug.augmenter.word as naw
# from transformers import MistralForCausalLM, MistralForSequenceClassification
import torch
from torch.autograd import Function
import torch.nn as nn
import numpy as np
from functools import partial
from peft import (
    # get_peft_model,
    # PeftType,
    # TaskType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    LoraConfig)
import os
import re

pad_to_multiple_of = 16
max_length = 512



def get_next_folder_name(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        return os.path.join(base_path, "model01")

    # List all items in the base directory
    items = os.listdir(base_path)

    # Filter out items that are not directories or don't match the pattern "modelXX"
    dir_pattern = re.compile(r"model(\d+)")
    model_dirs = [item for item in items if os.path.isdir(os.path.join(base_path, item)) and dir_pattern.match(item)]

    if not model_dirs:
        return os.path.join(base_path, "model01")

    # Extract numbers and find the maximum
    max_number = max(int(dir_pattern.match(d).group(1)) for d in model_dirs)

    # Create the next folder name
    next_folder_name = f"model{str(max_number + 1).zfill(2)}"
    return os.path.join(base_path, next_folder_name)

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
        topic = self.df.topics[idx]
        return text, label, score,topic

def collate_factory(data,tokenizer,IsTrain,prompt,prompt_mask):
    # prompt should be encoded
    if IsTrain:
        text, label, score, topic = zip(*data)
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
        return input_ids,attention_mask, torch.tensor(label, dtype=torch.float32), torch.tensor(score, dtype=torch.float32),torch.tensor(topic,dtype=torch.long)
    else:
        return input_ids,attention_mask

collate_fn = partial(collate_factory,IsTrain=True)
collate_inf = partial(collate_factory,IsTrain=False)


#############
### Model ###
#############
class GradientReversalFunction(Function):
    
    @staticmethod
    def forward(ctx, x):
        return x.to(dtype=torch.float32)
    
    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -1,)

class GradientReversalLayer(nn.Module):
    def forward(self, x):
        return GradientReversalFunction.apply(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim, dropout_prob):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_prob)]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            ])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        self.grad_rev = GradientReversalLayer()
    
    def forward(self, x):
        return self.layers(self.grad_rev(x))
        
class DAT(object):
    def __init__(self,topicModel,beta_):
        self.topicModel = topicModel
        self.beta_ = beta_
        self.loss_topic = torch.nn.CrossEntropyLoss()
    def get_DAT_loss(self,hidden,topic):
        logits = self.topicModel(hidden) # n,C
        return self.beta_ * self.loss_topic(logits,topic)
        
class LM(DAT):
    def __init__(self, model,tokenizer,num_virtual_tokens,alpha,config_type,topicModel,beta_):
        super().__init__(topicModel,beta_)
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
    
    def get_loss(self,input_ids,attention_mask,label,score,topic,*args,**kwargs):
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
        return loss_bce + self.alpha_ * loss_lm + self.get_DAT_loss(out.hidden_states.mean(dim=1),topic)

class Classification(DAT):
    def __init__(self,model, alpha,topicModel,beta_):
        super().__init__(topicModel,beta_)
        self.model = model
        self.alpha_ = alpha
        self.loss_bce_ = torch.nn.BCEWithLogitsLoss()
        self.loss_score_ = lambda y,yhat: torch.sum((y!=-1.)*torch.abs(yhat-y))/torch.sum((y!=-1.)+0.01)
        
    def predict(self,input_ids,attention_mask,*args,**kwargs):
        with torch.no_grad():
            out = self.model(input_ids,attention_mask,*args,**kwargs).logits
            return out[:,0]
    
    def get_loss(self,input_ids,attention_mask,label,score,topic,*args,**kwargs):
        out = self.model(input_ids,attention_mask,*args,**kwargs).logits
        return self.loss_bce_(out[:,0], label) + self.alpha_ * self.loss_score_(score,out[:,1]) + self.get_DAT_loss(out.hidden_states.mean(dim=1),topic)

def get_random_config(config_type,pred_type,TARGET_MODEL):
    if config_type == 'prefix':
        config_class = PrefixTuningConfig
        config_kwargs = dict(task_type = 'CAUSAL_LM' if pred_type =='LM' else "SEQ_CLS",
                            num_virtual_tokens = np.random.choice([16,24,32]), 
                            encoder_hidden_size = np.random.choice([2048,4096,6144]), 
                            prefix_projection = np.random.choice([True,False]),
                            )
        if TARGET_MODEL == "mistralai/Mistral-7B-v0.1":
            config_kwargs.update({'token_dim':1024,'num_attention_heads':8})
    elif config_type == 'prompt_encoder':
        config_class = PromptEncoderConfig
        config_kwargs = dict(task_type = 'CAUSAL_LM' if pred_type =='LM' else "SEQ_CLS", 
                            num_virtual_tokens = np.random.choice([16,24,32]), 
                            encoder_hidden_size = np.random.choice([2048,4096,6144]),
                            encoder_dropout = np.random.rand()*0.25,
                            encoder_num_layers = np.random.choice([1,2,3]),
                            encoder_reparameterization_type = np.random.choice(['MLP','LSTM'])
                            )
    elif config_type == 'prompt_txt':
        config_class = PromptTuningConfig
        config_kwargs = dict(task_type = 'CAUSAL_LM' if pred_type =='LM' else "SEQ_CLS",
                            prompt_tuning_init='TEXT', 
                            num_virtual_tokens = np.random.choice([24,48,64]),
                            prompt_tuning_init_text="Assessing the origin of an essay, whether it is a product of artificial intelligence, specifically a large language model (LLM), or the creative endeavor of a human student, involves a nuanced evaluation of various elements including the writing style, complexity of ideas, and the nuances in expression, a task that challenges us to discern if the piece reflects the unique perspective and contextual understanding characteristic of human thought, or if it embodies the sophisticated yet distinct patterns of language and content generation typical of advanced AI systems.",
                            tokenizer_name_or_path=TARGET_MODEL)
    elif config_type == 'LoRA':
        config_class = LoraConfig
        config_kwargs = dict(r=np.random.choice([32,64,128]),
                            lora_alpha = 16,
                            lora_dropout = np.random.rand()*0.25, 
                            bias=np.random.choice(['none', 'all' , 'lora_only' ]),
                            target_modules = ["q_proj","k_proj", "v_proj","o_proj"] if np.random.rand()<0.5 else ["q_proj","k_proj", "v_proj","o_proj","gate_proj", "up_proj", "down_proj" ]
                            )
    return config_class,config_kwargs

config_map = {'prefix':PrefixTuningConfig,
                'prompt_encoder':PromptEncoderConfig,
                'prompt_txt': PromptTuningConfig,
                'LoRA': LoraConfig,}

def save_config(TARGET_MODEL, pred_type, config_type, epochs, alpha, aug_kwargs, config_kwargs):
    config_dict = {
        "TARGET_MODEL": TARGET_MODEL,
        "pred_type": pred_type,
        "config_type": config_type,
        "epochs": epochs,
        "alpha": alpha,
        "aug_kwargs": aug_kwargs,
        "config_class": config_type,
        "config_kwargs": config_kwargs
    }
    return config_dict

def load_config(config_dict):
    TARGET_MODEL = config_dict["TARGET_MODEL"]
    pred_type = config_dict["pred_type"]
    config_type = config_dict["config_type"]
    epochs = config_dict["epochs"]
    alpha = config_dict["alpha"]
    aug_kwargs = config_dict["aug_kwargs"]
    config_class = config_map[config_dict["config_class"]]
    config_kwargs = config_dict["config_kwargs"]

    return TARGET_MODEL, pred_type, config_type, epochs, alpha, aug_kwargs, config_class, config_kwargs
