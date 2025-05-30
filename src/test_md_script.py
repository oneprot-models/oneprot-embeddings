import torch
import sys
import os
from pathlib import Path
import logging


from src.models.components.md_encoder import TrajectoryEncoder
from src.data.datasets.md_dataset import MDDataset
#from mdgen.wrapper import NewMDGenWrapper

dataset = MDDataset(
            split="test",
            num_frames=10,  # Use fewer frames for testing
            crop=128,       # Use smaller crop size for testing
            atlas=True      # Assuming this is needed for your dataset
        )

latent_dim=21
sample_idx = 0
sample_name = dataset[sample_idx]
seq_input, mdgen_batch, modality, sequences = dataset.collate_fn([sample_name])

encoder = TrajectoryEncoder(
            output_dim=1024,
            hidden_size=21,  # Smaller size for testing
            num_layers=4,     # Fewer layers for testing
            num_heads=8,
            pretrained=True,
            frozen=False,
            proj_type='mlp',
            
        )

encoder.eval()

outputs=encoder(mdgen_batch['latents'],0,**mdgen_batch['model_kwargs'])