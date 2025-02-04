import torch
import hydra
from omegaconf import OmegaConf
from huggingface_hub import  HfApi, hf_hub_download
import sys
import os
import h5py
from torch_geometric.data import Batch
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # assuming that you are running this script from the oneprot repo, can be any other path
from src.models.oneprot_module import OneProtLitModule
from src.data.utils.struct_graph_utils import protein_to_graph


#Load the config file and read it off

config_path = hf_hub_download(
        repo_id="sealinka/oneprot",
        filename="config.yaml",
    )

with open(config_path, 'r') as f:
    cfg = OmegaConf.load(f)

# Prepare components dictionary from config
components = {
        'sequence': hydra.utils.instantiate(cfg.model.components.sequence),
        'struct_token': hydra.utils.instantiate(cfg.model.components.struct_token),
        'struct_graph': hydra.utils.instantiate(cfg.model.components.struct_graph),
        'pocket': hydra.utils.instantiate(cfg.model.components.pocket),
        'text': hydra.utils.instantiate(cfg.model.components.text)
    }

# Load the model checkpoint

checkpoint_path = hf_hub_download(
        repo_id="sealinka/oneprot",
        filename="pytorch_model.bin",
        repo_type="model"
    )

# Create model instance and load the checkpoint

model = OneProtLitModule(
        components=components,
        optimizer=None,
        loss_fn=cfg.model.loss_fn,
        local_loss=cfg.model.local_loss,
        gather_with_grad=cfg.model.gather_with_grad,
        use_l1_regularization=cfg.model.use_l1_regularization,
        train_on_all_modalities_after_step=cfg.model.train_on_all_modalities_after_step,
        use_seqsim=cfg.model.use_seqsim
    )

state_dict = torch.load(checkpoint_path)
model_state_dict = model.state_dict()
model.load_state_dict(state_dict, strict=True)

# Define the tokenisers

tokenizers = {
        'sequence': "facebook/esm2_t33_650M_UR50D",
        'struct_token': "facebook/esm2_t33_650M_UR50D",
        'text': "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    }

loaded_tokenizers = {}
for modality, tokenizer_name in tokenizers.items():
    tokenizer = AutoTokenizer.from_pretrained(tokenizers[modality])
    loaded_tokenizers[modality] = tokenizer

# Get example embeddings for each modality


##########################sequence##############################

modality = "sequence"
 
file_path = hf_hub_download(
    repo_id="sealinka/oneprot",
    filename="data_examples/sequence_example.txt",
    repo_type="model"  # or "dataset"
)

with open(file_path, 'r') as file:
    input_sequence = file.read().strip()

input_tensor = loaded_tokenizers[modality](input_sequence, return_tensors="pt")["input_ids"]
output = model.network[modality](input_tensor)
print(f"Output for modality '{modality}': {output}")


###########################text#################################

modality = "text"

file_path = hf_hub_download(
    repo_id="sealinka/oneprot",
    filename="data_examples/text_example.txt",
    repo_type="model"  # or "dataset"
)

with open(file_path, 'r') as file:    
    input_text = file.read().strip()

input_tensor = loaded_tokenizers[modality](input_text, return_tensors="pt")["input_ids"]
output = model.network[modality](input_tensor)
print(f"Output for modality '{modality}': {output}")


#####################tokenized structure########################

modality = "struct_token" 

file_path = hf_hub_download(
    repo_id="sealinka/oneprot",
    filename="data_examples/struct_token_example.txt",
    repo_type="model"  # or "dataset"
)

with open(file_path, 'r') as file:
    input_struct_token = file.read().strip()

input_struct_token = "".join([s.replace("#", "") for s in input_struct_token])
input_tensor = loaded_tokenizers[modality](input_struct_token, return_tensors="pt")["input_ids"]
output = model.network[modality](input_tensor)
print(f"Output for modality '{modality}': {output}")


#####################graph structure############################

modality = "struct_graph"  
file_path = hf_hub_download(
    repo_id="sealinka/oneprot",
    filename="data_examples/seqstruc_example.h5",
    repo_type="model"  # or "dataset"
)

with h5py.File(file_path, 'r') as file:
    input_struct_graph=[protein_to_graph('E6Y2X0', file_path, 'non_pdb', 'A', pockets=False)]
    input_struct_graph = Batch.from_data_list(input_struct_graph)
    output=model.network[modality](input_struct_graph)
print(f"Output for modality '{modality}': {output}")


##########################pocket################################


modality = "pocket"  # Replace with the desired modality

file_path = hf_hub_download(
    repo_id="sealinka/oneprot",
    filename="data_examples/pocket_example.h5",
    repo_type="model"  # or "dataset"
)

with h5py.File(file_path, 'r') as file:
    input_pocket=[protein_to_graph('E6Y2X0', file_path, 'non_pdb', 'A', pockets=True)]
    input_pocket = Batch.from_data_list(input_pocket)
    output=model.network[modality](input_pocket)
    
print(f"Output for modality '{modality}': {output}")
