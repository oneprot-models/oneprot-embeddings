import os
import logging
from typing import List, Any
import shutil

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModel, EsmForMaskedLM, EsmTokenizer
import hydra
from omegaconf import DictConfig, OmegaConf
import pyrootutils
import ast
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, LogitsConfig, LogitsOutput,ESMProteinError
from esm.sdk import batch_executor


data=pd.read_csv('/p/scratch/hai_oneprot/bazarova1/csv_files/TopEnzyme/Comb826_train_seq_long.csv')
model=torch.load('/p/scratch/hai_oneprot/huggingface/models/esm3_sm_open_v1_full.pth',weights_only=False)
k=18

for i in range(k,len(data)):
    print(i)
    sequence=data['sequence'][i]
    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)
    output = model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
    all_embeddings=torch.mean(output.embeddings,dim=1).squeeze(1)
    torch.save({"embeddings": all_embeddings.cpu(), "labels_fitness": torch.tensor([data['label/fitness'][i]]).cpu()},f'/p/scratch/hai_oneprot/bazarova1/embeddings/esm3/TopEnzyme_long/train/single_embs/embeddings_rank0_batch{i}.pt')
