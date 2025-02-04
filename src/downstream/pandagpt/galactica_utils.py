import hydra
from pytorch_lightning import LightningModule, LightningDataModule
#from src.data.components.datasets import text_collate_fn
#from src.data.datasets.text_dataset import text_collate_fn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import torch.nn as nn
import torch
from torch.nn.utils import rnn
from torch.utils.data import DataLoader
import os


from transformers import StoppingCriteria, StoppingCriteriaList

import torch
from torch.nn.utils import rnn
PROMPT_START = '### Human: <protein>'

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()
        if stop_count >= self.ENCOUNTERS:
            return True
        return False
    

def define_prompt_wrap(modality_embs, input_ids, target_ids, attention_mask, model, tokenizer_local):
    local_rank = int(os.environ.get('SLURM_LOCALID', '0'))
    device = torch.device(f'cuda:{local_rank}')
    batch_size = modality_embs.shape[0]
    p_before = PROMPT_START
    p_before_tokens = tokenizer_local(p_before, return_tensors="pt", add_special_tokens=False).to(device)
    # i cant explain what happens here and why
    p_before_embeds = model.module.model.model.decoder.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
    p_after_embeds = model.module.model.model.decoder.embed_tokens(input_ids).expand(batch_size, -1, -1)
    bos = torch.ones([batch_size, 1, 1],
                        dtype=p_before_tokens.input_ids.dtype,
                        device=p_before_tokens.input_ids.device) * tokenizer_local.bos_token_id 
    
    bos_embeds = model.module.model.model.decoder.embed_tokens(bos).squeeze(2)

    inputs_embeds = torch.cat([bos_embeds, p_before_embeds, modality_embs.unsqueeze(1), p_after_embeds], dim=1)
    # create targets
    empty_targets = (
        torch.ones([batch_size, 1+p_before_embeds.size()[1]+1], # 1 (bos) + s1 + 1 (image vector)
                    dtype=torch.long).fill_(-100)  
    ).to(device) # bsz x (1 + s1 + 1)
    targets = torch.cat([empty_targets, target_ids], dim=1) # bsz x (1 + s1 + 1 + s2)
    assert inputs_embeds.size()[1] == targets.size()[1]

    atts_prefix = torch.ones([batch_size, 1+p_before_embeds.size()[1]+1], dtype=torch.long).to(device) # bsz x (1 + s1 +1)
    attention_mask = torch.cat([atts_prefix, attention_mask], dim=1)
    assert attention_mask.size() == targets.size() # bsz x (1 + s1 + 1 + s2)
    return inputs_embeds, targets, attention_mask 

def build_one_instance(tokenizer, conversation):
        text_list = []
        turn_num = len(conversation)
        input_ids, target_ids = [], []
        for i in range(turn_num):
            turn = conversation[i]
            role = turn['from']
            if i == 0: # the first human turn
                assert role == 'human'
                text = '</protein> ' + turn['value'] + '\n### Assistant:'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += [-100]*len(one_input_id) # do not perform loss regression on human prompt
            else:
                if role == 'human':
                    text = 'Human: ' + turn['value'] + '\n### Assistant:'
                    one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                    input_ids += one_input_id
                    target_ids += [-100]*len(one_input_id)
                elif role == 'gpt':
                    text = turn['value'] + '\n###'
                    one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                    input_ids += one_input_id
                    target_ids += one_input_id
                else:
                    raise Exception('Wrong Role!!!')
            text_list.append(text)
            assert len(input_ids) == len(target_ids)
        return text_list, input_ids, target_ids

@torch.no_grad()
def encode_modality(modality_inputs, modality_encoder, modality_proj):
    modality_embs = modality_encoder(modality_inputs)
    modality_embs = modality_proj.module(modality_embs)
    return modality_embs

def process_batch_instance(tokenizer, batch_of_conversations, max_tgt_len):
    local_rank = int(os.environ.get('SLURM_LOCALID', '0'))
    device = torch.device(f'cuda:{local_rank}')
    
    batch_input_ids, batch_target_ids = [], []
    for conversation in batch_of_conversations:
        _, one_input_ids, one_target_ids = build_one_instance(tokenizer, conversation)
        batch_input_ids.append(torch.LongTensor(one_input_ids).to(device))
        batch_target_ids.append(torch.LongTensor(one_target_ids).to(device))
    
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    
    input_ids = input_ids[:,:max_tgt_len]
    target_ids = target_ids[:,:max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    
    return input_ids, target_ids, attention_mask.long()

def prepare_generation_embedding(inputs, model, tokenizer_local,emb):
    local_rank = int(os.environ.get('SLURM_LOCALID', '0'))
    device = torch.device(f'cuda:{local_rank}')
    #batch_size = modality_embs.shape[0]
    prompt = inputs['prompt']
    if len(inputs[emb]) == 1:
        feature_embeds = inputs[emb][0]
    else:
        raise NotImplementedError

    batch_size = feature_embeds.shape[0]
    p_before = PROMPT_START
    p_before_tokens = tokenizer_local(p_before, 
        return_tensors="pt", add_special_tokens=False).to(device)
    p_before_embeds = model.module.model.model.decoder.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
    text = '</protein> ' + prompt + '\n### Assistant:'
    p_after_tokens = tokenizer_local(text, add_special_tokens=False, return_tensors='pt').to(device)
    p_after_embeds = model.module.model.model.decoder.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
    bos = torch.ones([batch_size, 1, 1],
                        dtype=p_before_tokens.input_ids.dtype,
                        device=p_before_tokens.input_ids.device) * tokenizer_local.bos_token_id # bsz x 1
    bos_embeds = model.module.model.model.decoder.embed_tokens(bos).squeeze(2).to(device) # bsz x 1 x embed_dim
    inputs_embeds = torch.cat([bos_embeds, p_before_embeds, feature_embeds.unsqueeze(1), p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
    return inputs_embeds