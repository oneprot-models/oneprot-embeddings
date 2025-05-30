#!/usr/bin/env python3
import hydra
from omegaconf import DictConfig
import os
import sys

# Path setup
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

@hydra.main(config_path="../configs/experiment", config_name="train.yaml")
def main(cfg: DictConfig):
    print("Configuration loaded:", cfg)
    
    # Import your training function here
    from src.train_oneprot import train_function  # Assuming your main training function is called train_function
    
    # Run with the Hydra config
    return train_function(cfg)

if __name__ == "__main__":
    main()