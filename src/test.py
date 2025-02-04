# import torch
# from omegaconf import OmegaConf
# import hydra
# import pytorch_lightning as pl

# config_path='/p/project1/hai_oneprot/bazarova1/oneprot-refined/logs/train/runs/2024-11-05_16-37-44/config.yaml'
# ckpt_path = '/p/scratch/hai_oneprot/checkpoints_refined_111024/2024-11-05_19-20-45/epoch_043_28400.ckpt'

# # Load the configuration
# cfg = OmegaConf.load(config_path)

# # Instantiate the model
# model = hydra.utils.instantiate(cfg.model)

# # Load the entire model from checkpoint
# model = model.__class__.load_from_checkpoint(
#     checkpoint_path=ckpt_path,
#     **OmegaConf.to_container(cfg.model, resolve=True)
# )

# # Set to evaluation mode
# model.eval()

import hydra
from omegaconf import OmegaConf
import sys
import os
import torch
#from src.eval import create_dataloader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.eval import create_dataloader, encode_inputs
import numpy as np


@hydra.main(config_path="/p/project1/hai_oneprot/bazarova1/oneprot-refined/logs/train/runs/2024-11-05_16-37-44/", config_name="config.yaml")
def main(cfg):
    ckpt_path = '/p/scratch/hai_oneprot/checkpoints_refined_111024/2024-11-05_19-20-45/epoch_043_28400.ckpt'
    print(OmegaConf.to_yaml(cfg))  # Print the configuration to verify it
    model = hydra.utils.instantiate(cfg.model)
    #print(model)

    model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    model.cuda()
    model.print_init_args()
    #checkpoint = torch.load(ckpt_path, map_location='cpu')

    # Load the state dict into the model
    #model.load_state_dict(checkpoint['state_dict'])

    # Optionally, set the model to evaluation mode
    model.eval()

    # dataloader = create_dataloader(cfg)

    # available_modalities = list(next(iter(dataloader)).keys())
    
    # all_embeddings = {modality: [] for modality in available_modalities if modality != 'ids'}
    
    # Process dataset
    # for batch in dataloader:
    #     embedded_tokens = encode_inputs(model, batch)
    #     for modality, embeddings in embedded_tokens.items():
    #         if modality != 'ids':
    #             all_embeddings[modality].append(embeddings.cpu().numpy())
    
    # # Concatenate embeddings
    # for modality in all_embeddings:
    #     all_embeddings[modality] = np.concatenate(all_embeddings[modality], axis=0)
    
    # print(all_embeddings['pocket'].shape,all_embeddings['struct_graph'].shape,all_embeddings['struct_token'].shape,all_embeddings['sequence'].shape,all_embeddings['text'].shape)
    # print(all_embeddings['pocket'])

if __name__ == "__main__":
    main()