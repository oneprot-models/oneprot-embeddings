import torch
import hydra
from omegaconf import OmegaConf
from huggingface_hub import  HfApi, hf_hub_download
import sys
import os
import h5py
from torch_geometric.data import Batch
from transformers import AutoTokenizer
import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # assuming that you are running this script from the oneprot repo, can be any other path

from src.models.oneprot_module import OneProtLitModule
from src.data.utils.struct_graph_utils import protein_to_graph

os.environ['RANK']='0'
os.environ['WORLD_SIZE']='1'

config_path = "/p/project1/hai_oneprot/bazarova1/oneprot-panda/configs/collect_embeddings.yaml"
cfg = OmegaConf.load(config_path)

cfg.model = cfg.models[0]

from src.collect_embeddings import load_custom_model
model = load_custom_model(cfg)
import pandas as pd
embeddings_dict = {}
tokenizers = {
        'sequence': "facebook/esm2_t33_650M_UR50D",
        'struct_token': "facebook/esm2_t33_650M_UR50D",
        'text': "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    }

loaded_tokenizers = {}
for modality, tokenizer_name in tokenizers.items():
    tokenizer = AutoTokenizer.from_pretrained(tokenizers[modality])
    if modality=='struct_token':
      new_tokens = ['p', 'y', 'n', 'w', 'r', 'q', 'h', 'g', 'd', 'l', 'v', 't', 'm', 'f', 's', 'a', 'e', 'i', 'k', 'c','#']
      tokenizer.add_tokens(new_tokens)
    loaded_tokenizers[modality] = tokenizer

modality='sequence'

csv_path = "/p/project1/hai_oneprot/bazarova1/mdgen/splits/atlas_train.csv"
df = pd.read_csv(csv_path)

for idx, row in df.iterrows():
    name = row['name']
    seq = row['seqres']
    input_tensor = loaded_tokenizers[modality](seq, return_tensors="pt")["input_ids"]
    output=model(input_tensor)
    embeddings_dict[name]=output
    print(idx)


# Save to file
torch.save(embeddings_dict, "/p/data1/profound_data/oneprot_trans_atlas_train.pt")