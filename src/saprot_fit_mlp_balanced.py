from typing import Dict
import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, RichProgressBar, ModelCheckpoint
from pytorch_lightning import LightningDataModule
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from itertools import product
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
import os
import sys
import numpy as np

import random

def set_partial_seed(seed, deterministic_level="medium"):
    """
    Set seeds with different levels of determinism
    
    Args:
        seed: Random seed
        deterministic_level: "low", "medium", or "high"
    """
    
    if deterministic_level == "high":
        # Your current approach - fully deterministic
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        pl.seed_everything(seed, workers=True)
    
    elif deterministic_level == "medium":
        # Fix initialization but allow some training randomness
        random.seed(seed)
        np.random.seed(seed) 
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Allow CUDNN to be non-deterministic for better performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        pl.seed_everything(seed, workers=False)  # Don't seed workers
    
    elif deterministic_level == "low":
        # Only fix weight initialization
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Keep everything else random
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#pl.seed_everything(cfg.seed, workers=True)
# Get the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory and its parent to the Python path
sys.path.insert(0, current_dir)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.downstream import save_results_to_csv, load_data, count_f1_max

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, targets):
        self.embeddings = embeddings
        self.targets = targets

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx]

class EmbeddingDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.data = {}

    def setup(self, stage=None):
        all_inputs = load_data(self.cfg)
        for partition in self.cfg.evaluate_on:
            embeddings = all_inputs[f"{partition}_emb"]
            targets = all_inputs[f"{partition}_target"]
            
            # Balance training data for merged binary tasks
            if partition == "train" and self.cfg.task_name in ["ASD_merged_pocket_binary", "ASD_merged_pocket_sequence_binary","ASD_merged_pocket_binary_text","ASD_merged_pocket_sequence_binary_text","ASD_merged_pocket_binary_comp", "ASD_merged_pocket_sequence_binary_comp","ASD_merged_pocket_binary_text_comp","ASD_merged_pocket_sequence_binary_text_comp"]:
                # Ensure targets is a tensor
                if not isinstance(targets, torch.Tensor):
                    targets = torch.from_numpy(targets)
                if not isinstance(embeddings, torch.Tensor):
                    embeddings = torch.from_numpy(embeddings)
                
                pos_mask = (targets == 1)
                neg_mask = (targets == 0)
                pos_idx = torch.where(pos_mask)[0]
                neg_idx = torch.where(neg_mask)[0]
                
                n_pos = len(pos_idx)
                n_neg = len(neg_idx)
                
                # Subsample the majority class to match minority
                if n_pos > n_neg:
                    # More positives than negatives - subsample positives
                    selected_pos = pos_idx[torch.randperm(n_pos)[:n_neg]]
                    selected_idx = torch.cat([selected_pos, neg_idx])
                else:
                    # More negatives than positives - subsample negatives
                    selected_neg = neg_idx[torch.randperm(n_neg)[:n_pos]]
                    selected_idx = torch.cat([pos_idx, selected_neg])
                
                # Shuffle the combined indices
                selected_idx = selected_idx[torch.randperm(len(selected_idx))]
                
                embeddings = embeddings[selected_idx]
                targets = targets[selected_idx]
                
                print(f"Balanced {partition} set: {len(selected_idx)} samples (pos: {(targets==1).sum()}, neg: {(targets==0).sum()})")
            
            self.data[partition] = EmbeddingDataset(embeddings, targets)

    def train_dataloader(self):
        return DataLoader(
            self.data["train"], batch_size=self.cfg.model.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.data["valid"], batch_size=self.cfg.model.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data["test"], batch_size=self.cfg.model.batch_size)

# class MLP(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dims=[256], dropout_rate=0.0, 
#                  use_batch_norm=False, use_layer_norm=False, activation='relu', 
#                  use_residual=False,num_classes=4):
#         super().__init__()
#         self.use_residual = use_residual
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.pos_embedding = nn.Parameter(
#             torch.randn(output_dim, input_dim) * 0.02
#         )
#         layers = []
#         in_features = input_dim


#         for i, hidden_dim in enumerate(hidden_dims):
#             layers.append(nn.Linear(in_features, hidden_dim))
            
#             if use_batch_norm:
#                 layers.append(nn.BatchNorm1d(hidden_dim))
#             elif use_layer_norm:
#                 layers.append(nn.LayerNorm(hidden_dim))
            
#             if activation == 'relu':
#                 layers.append(nn.ReLU())
#             elif activation == 'gelu':
#                 layers.append(nn.GELU())
#             elif activation == 'leaky_relu':
#                 layers.append(nn.LeakyReLU())
            
#             if dropout_rate > 0:
#                 layers.append(nn.Dropout(dropout_rate))
            
#             in_features = hidden_dim

#         self.hidden_layers = nn.Sequential(*layers)
#         #self.hidden_layers = nn.ModuleList(layers)
#         #self.output_layer = nn.Linear(in_features, output_dim)
#         self.output_layer = nn.Linear(in_features, num_classes)


#         # Add a projection layer for residual connections if needed
#         if self.use_residual and input_dim != hidden_dims[-1]:
#             self.residual_projection = nn.Linear(input_dim, hidden_dims[-1])
#         else:
#             self.residual_projection = None

#    def forward(self, x):
        # batch_size=x.shape[0]

        # x_expanded = x.unsqueeze(1).expand(batch_size, self.output_dim, -1)
        # pos_emb = self.pos_embedding.unsqueeze(0).expand(batch_size, -1, -1)

        # x=x_expanded + pos_emb
        # if self.use_residual:
        #     residual = x

        # # for layer in self.hidden_layers:
        # #     x = layer(x)
        # x=self.hidden_layers(x)

        # if self.use_residual:
        #     if self.residual_projection:
        #         residual = self.residual_projection(residual)
        #     x = x + residual
        
        # x = self.output_layer(x)
        
        # # Reshape to [batch, seq_len, num_classes]
        # batch_size = x.shape[0]
        # x = x.view(batch_size, self.output_dim, -1)

        #return self.output_layer(x)
#        return x

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256], dropout_rate=0.0, 
                 use_batch_norm=False, use_layer_norm=False, activation='relu', 
                 use_residual=False):
        super().__init__()
        self.use_residual = use_residual
        self.input_dim = input_dim

        layers = []
        in_features = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_features, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            in_features = hidden_dim

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(in_features, output_dim)

        # Add a projection layer for residual connections if needed
        if self.use_residual and input_dim != hidden_dims[-1]:
            self.residual_projection = nn.Linear(input_dim, hidden_dims[-1])
        else:
            self.residual_projection = None

    def forward(self, x):
        if self.use_residual:
            residual = x

        for layer in self.hidden_layers:
            x = layer(x)

        if self.use_residual:
            if self.residual_projection:
                residual = self.residual_projection(residual)
            x = x + residual

        return self.output_layer(x)

class EvaluationModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        if cfg.model_type == "esm2" or cfg.model_type == "saprot":
            input_dim = 1280
        elif cfg.model_type in ["oneprot_9"]:
            input_dim = 256
        elif cfg.model_type in ["oneprot_15", "oneprot_16"]:
            input_dim = 1280
        elif "esm3" in cfg.model_type:
            input_dim=1536
        else:
            input_dim = 1024 #normal
            if cfg.model_type == "esmIF-08-31-2025":
                input_dim = 512
            elif cfg.model_type == "embeddings_saprot":
                input_dim = 1280
            #input_dim = 512 #esmIF
            #input_dim=1280 #openfold
        if self.cfg.task_name in ["HumanPPI", "ASD_pocket_sequence100", "ASD_pockets_sequence", "Kinase_combined", "merged_pocket_sequence","ASD_pockets_sequence_binary","merged_pocket_sequence_binary","ASD_merged_pocket_sequence_binary","ASD_merged_pocket_binary_text","ASD_merged_pocket_sequence_binary_comp","ASD_merged_pocket_binary_text_comp"]:
            input_dim = input_dim * 2
        if self.cfg.task_name in ["ASD_merged_pocket_sequence_binary_text","ASD_merged_pocket_sequence_binary_text_comp"]:
            input_dim = input_dim * 3

        # Determine output_dim based on task_name

        if self.cfg.task_name in ["MetalIonBinding", "DeepLoc2", "HumanPPI", "ThermoStability","Thermostability","Kinase_combined","Kinase_pocket","ASD_pockets_binary","ASD_pockets_sequence_binary","merged_pocket_sequence_binary","merged_pocket_binary","ASD_merged_pocket_binary","ASD_merged_pocket_sequence_binary","ASD_merged_pocket_binary_text","ASD_merged_pocket_sequence_binary_text","ASD_merged_pocket_binary_comp","ASD_merged_pocket_sequence_binary_comp","ASD_merged_pocket_binary_text_comp","ASD_merged_pocket_sequence_binary_text_comp"]:
            output_dim = 1
        elif self.cfg.task_name == "EC":
            output_dim = 585
        elif self.cfg.task_name == "GO-BP":
            output_dim = 1943
        elif self.cfg.task_name == "GO-MF":
            output_dim = 489
        elif self.cfg.task_name == "GO-CC":
            output_dim = 320
        elif self.cfg.task_name == "DeepLoc10":
            output_dim = 10
        elif self.cfg.task_name == "ASD_pockets" or self.cfg.task_name=="ASD_pockets_sequence" or self.cfg.task_name=="merged_pocket" or self.cfg.task_name=="merged_pocket_sequence":
            output_dim = 3
        elif self.cfg.task_name == "TopEnzyme":
            output_dim = 826
        elif self.cfg.task_name == "ASD":
            output_dim = 1024*5
        elif self.cfg.task_name == "PL8":
            output_dim = 3
        elif self.cfg.task_name=="ASD_pockets100" or self.cfg.task_name=="ASD_pocket_sequence100":
            output_dim=100
        else:
            raise ValueError(f"Unknown task_name: {self.cfg.task_name}")

        self.model = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=cfg.model.hidden_dims,
            dropout_rate=cfg.model.dropout_rate,
            use_batch_norm=cfg.model.use_batch_norm,
            use_layer_norm=cfg.model.use_layer_norm,
            activation=cfg.model.activation,
            use_residual=cfg.model.use_residual
        )

        # def masked_bce_with_logits(logits, targets, pad_value=-1.0):
        #     mask = targets != pad_value
        #     logits = logits[mask]
        #     targets = targets[mask]
        # return F.binary_cross_entropy_with_logits(logits, targets)

        # Set loss function based on task_name
        if self.cfg.task_name in ["MetalIonBinding", "DeepLoc2", "HumanPPI", "EC", "GO-BP", "GO-MF", "GO-CC", "PL8","ASD_pockets100","ASD_pocket_sequence100","ASD_pockets_binary","ASD_pockets_sequence_binary","merged_pocket_sequence_binary","merged_pocket_binary","ASD_merged_pocket_binary","ASD_merged_pocket_sequence_binary","ASD_merged_pocket_binary_text","ASD_merged_pocket_sequence_binary_text","ASD_merged_pocket_binary_comp","ASD_merged_pocket_sequence_binary_comp","ASD_merged_pocket_binary_text_comp","ASD_merged_pocket_sequence_binary_text_comp"]:
            self.loss_fn = F.binary_cross_entropy_with_logits
        elif self.cfg.task_name in ["ThermoStability","Thermostability"]:
            self.loss_fn = F.mse_loss
        elif self.cfg.task_name == "ASD":
            self.loss_fn = None
        else:  # multi_class
            self.loss_fn = F.cross_entropy

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.cfg.task_name == "ASD":

            y_hat = y_hat.view(-1, 1024, 5)
            y=y.long()
            #print(y_hat.shape," shape y hat!!!!!", flush=True)
            # batch_size, seq_len, num_classes = y_hat.shape
            # y_hat_flat = y_hat.reshape(-1, num_classes)  # [batch*seq_len, num_classes]
            # y_flat = y.reshape(-1)  # [batch*seq_len]
        
            #loss = F.cross_entropy(y_hat_flat, y_flat.long(), ignore_index=-100)
            loss = F.cross_entropy(y_hat.permute(0, 2, 1), y,ignore_index=-100)
        elif self.cfg.task_name in ["MetalIonBinding", "DeepLoc2", "ThermoStability", "HumanPPI","Kinase_combined","Kinase_pocket","ASD_pockets_binary","ASD_pockets_sequence_binary","merged_pocket_sequence_binary","merged_pocket_binary","ASD_merged_pocket_binary","ASD_merged_pocket_sequence_binary","ASD_merged_pocket_binary_text","ASD_merged_pocket_sequence_binary_text","ASD_merged_pocket_binary_comp","ASD_merged_pocket_sequence_binary_comp","ASD_merged_pocket_binary_text_comp","ASD_merged_pocket_sequence_binary_text_comp"]:
            y_hat = y_hat.squeeze(1)
            y = y.float()
            loss = self.loss_fn(y_hat, y)
        elif self.cfg.task_name in ["EC", "GO-BP", "GO-MF", "GO-CC", "PL8"]:
            y_hat = y_hat.float()
            y = y.float()
            loss = self.loss_fn(y_hat, y)
        elif self.cfg.task_name in ["ASD_pockets100","ASD_pocket_sequence100"]:
            y_hat = y_hat.float()
            y = y.float()
            mask = y != -1
            y_hat = y_hat[mask]
            y = y[mask]
            loss = self.loss_fn(y_hat, y)
        else:
            loss = self.loss_fn(y_hat, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # if batch_idx == 0:
        #     print(f"y shape: {y.shape}")
        #     print(f"y min: {y.min()}, y max: {y.max()}")
        #     print(f"y unique values: {torch.unique(y)}")
        #     print(f"Number of invalid labels (<0): {(y < 0).sum()}")
        #     print(f"Number of invalid labels (>=3): {(y >= 3).sum()}")
    
        if self.cfg.task_name == "ASD":
            y_hat = y_hat.view(-1, 1024, 5)
            y = y.long()
            loss = F.cross_entropy(y_hat.permute(0, 2, 1), y,ignore_index=-100)
            #print(y_hat.shape," shape y hat!!!!!", flush=True)
        
            # batch_size, seq_len, num_classes = y_hat.shape
            # y_hat_flat = y_hat.reshape(-1, num_classes)  # [batch*seq_len, num_classes]
            # y_flat = y.reshape(-1)  # [batch*seq_len]
        
            # loss = F.cross_entropy(y_hat_flat, y_flat.long(), ignore_index=-100)
        elif self.cfg.task_name in ["MetalIonBinding", "DeepLoc2", "ThermoStability", "HumanPPI","Thermostability","Kinase_combined","Kinase_pocket","ASD_pockets_binary","ASD_pockets_sequence_binary","merged_pocket_sequence_binary","merged_pocket_binary","ASD_merged_pocket_binary","ASD_merged_pocket_sequence_binary","ASD_merged_pocket_binary_text","ASD_merged_pocket_sequence_binary_text","ASD_merged_pocket_binary_comp","ASD_merged_pocket_sequence_binary_comp","ASD_merged_pocket_binary_text_comp","ASD_merged_pocket_sequence_binary_text_comp"]:
            y_hat = y_hat.squeeze(1)
            y = y.float()
            loss = self.loss_fn(y_hat, y)
        elif self.cfg.task_name in ["EC", "GO-BP", "GO-MF", "GO-CC", "PL8"]:
            y_hat = y_hat.float()
            y = y.float()
            loss = self.loss_fn(y_hat, y)
        elif self.cfg.task_name in ["ASD_pockets100","ASD_pocket_sequence100"]:
            y_hat = y_hat.float()
            y = y.float()
            mask = y != -1
            y_hat = y_hat[mask]
            y = y[mask]
            loss = self.loss_fn(y_hat, y)
        else:
            loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)

    def on_validation_epoch_end(self):
        avg_val_loss = self.trainer.callback_metrics["val_loss"]
        self.log("val_loss_epoch", avg_val_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.model.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)

        if self.cfg.task_name == "ASD":
            # Reshape and get class predictions
            y_hat = y_hat.view(-1, 1024, 5)  # [batch, 1024, 3]
            preds = torch.argmax(y_hat, dim=-1)  # [batch, 1024] - class per position
            # if y_hat.dim() == 3:
            #     y_hat = y_hat.squeeze(0)
            #     y = y.squeeze(0)
        
            #preds = torch.argmax(y_hat, dim=-1)        
        elif self.cfg.task_name in ["MetalIonBinding", "DeepLoc2", "HumanPPI","Kinase_combined","Kinase_pocket","ASD_pockets_binary","ASD_pockets_sequence_binary","merged_pocket_sequence_binary","merged_pocket_binary","ASD_merged_pocket_binary","ASD_merged_pocket_sequence_binary","ASD_merged_pocket_binary_text","ASD_merged_pocket_sequence_binary_text","ASD_merged_pocket_binary_comp","ASD_merged_pocket_sequence_binary_comp","ASD_merged_pocket_binary_text_comp","ASD_merged_pocket_sequence_binary_text_comp"]:
            y_hat = y_hat.squeeze(1)
            preds = torch.sigmoid(y_hat)
        elif self.cfg.task_name in ["EC", "GO-BP", "GO-MF", "GO-CC", "PL8","ASD_pockets100","ASD_pocket_sequence100"]:
            preds = torch.sigmoid(y_hat)
        elif self.cfg.task_name in ["ThermoStability", "Thermostability"]:
            preds = y_hat.squeeze(1)
        else:  # multi_class
            preds = torch.softmax(y_hat, dim=1)
            preds = torch.argmax(preds, dim=1)


        return preds, y



def evaluate(cfg: DictConfig, data_module: EmbeddingDataModule) -> Dict:
    model = EvaluationModule(cfg)

    # Determine the accelerator and devices based on GPU availability
    if torch.cuda.is_available():
        accelerator = "gpu"
        fit_devices = 4
        predict_devices = 1
    else:
        accelerator = "cpu"
        fit_devices = 1
        predict_devices = 1

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.model.early_stopping_patience,
        mode="min",
        verbose=True,
        log_rank_zero_only=True,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_epoch",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best-checkpoint",
    )

    fit_trainer = pl.Trainer(
        max_epochs=cfg.model.max_epochs,
        accelerator=accelerator,
        devices=fit_devices,
        strategy="auto",
        callbacks=[early_stop_callback, RichProgressBar(), checkpoint_callback],
    )

    fit_trainer.fit(model, data_module)

    # Load the best model
    best_model_path = checkpoint_callback.best_model_path
    model = EvaluationModule.load_from_checkpoint(best_model_path)

    predict_trainer = pl.Trainer(
        accelerator=accelerator,
        devices=predict_devices,
        strategy="auto",
    )

    results = {}
    for partition in ["valid", "test"]:
        if partition == "valid":
            predictions = predict_trainer.predict(model, data_module.val_dataloader())
        else:
            predictions = predict_trainer.predict(
                model, getattr(data_module, f"{partition}_dataloader")()
            )

        if cfg.task_name in ["MetalIonBinding", "DeepLoc2", "HumanPPI","Kinase_combined","Kinase_pocket","ASD_pockets_binary","ASD_pockets_sequence_binary","merged_pocket_sequence_binary","merged_pocket_binary","ASD_merged_pocket_binary","ASD_merged_pocket_sequence_binary","ASD_merged_pocket_binary_text","ASD_merged_pocket_sequence_binary_text","ASD_merged_pocket_binary_comp","ASD_merged_pocket_sequence_binary_comp","ASD_merged_pocket_binary_text_comp","ASD_merged_pocket_sequence_binary_text_comp"]:
            y_pred = torch.cat([p[0] for p in predictions]).cpu().numpy()
            y_true = torch.cat([p[1] for p in predictions]).cpu().numpy()
            accuracy = accuracy_score(y_true, y_pred > 0.5)
            f1_micro = f1_score(y_true, y_pred > 0.5, average="micro")
            auc = roc_auc_score(y_true, y_pred)
            results[f"{partition}_accuracy"] = accuracy
            results[f"{partition}_f1_micro"] = f1_micro
            results[f"{partition}_auc"] = auc
            pred_binary = (y_pred > 0.5).astype(int)
            tp = int(((pred_binary == 1) & (y_true == 1)).sum())
            tn = int(((pred_binary == 0) & (y_true == 0)).sum())
            results[f"{partition}_tp"] = tp
            results[f"{partition}_tn"] = tn

        elif cfg.task_name in ["EC", "GO-BP", "GO-MF", "GO-CC", "PL8"]:
            y_pred = torch.cat([p[0] for p in predictions]).cpu()
            y_true = torch.cat([p[1] for p in predictions]).cpu()
            f1_max = count_f1_max(y_pred, y_true)
            results[f"{partition}_f1_max"] = f1_max
        elif cfg.task_name in ["ASD_pockets100","ASD_pocket_sequence100"]:
            y_pred = torch.cat([p[0] for p in predictions]).cpu()
            y_true = torch.cat([p[1] for p in predictions]).cpu()
            # Create mask to ignore padding tokens (-1)
            mask = y_true != -1
            #print(mask.shape, " mask shape")
            #print(y_pred.shape, y_true.shape, y_pred, " y pred shape")

            if y_pred.dim() == 2 and y_true.dim() == 2:
        # Remove completely masked samples (if any)
                valid_samples = mask.any(dim=1)
                y_pred = y_pred[valid_samples]
                y_true = y_true[valid_samples]
                mask = mask[valid_samples]
        
            if mask.all():
                # No masked labels - use count_f1_max directly
                f1_max = count_f1_max(y_pred, y_true)
            else:
                # Has masked labels - must filter them out first
                # count_f1_max expects ALL values to be valid (no -1)
                y_pred_flat = y_pred[mask]
                y_true_flat = y_true[mask]
            
            # Reshape to [1, num_valid_labels] for count_f1_max
                y_pred_2d = y_pred_flat.unsqueeze(0)
                y_true_2d = y_true_flat.unsqueeze(0)
            
                f1_max = count_f1_max(y_pred_2d, y_true_2d)
            results[f"{partition}_f1_max"] = f1_max
        elif cfg.task_name == "ASD":
            y_pred = torch.cat([p[0] for p in predictions]).cpu().numpy()  # Shape: [N, 1024]
            y_true = torch.cat([p[1] for p in predictions]).cpu().numpy()  # Shape: [N, 1024]
    
            # Flatten the arrays for per-position metrics
            y_pred_flat = y_pred.flatten()
            y_true_flat = y_true.flatten()
    
            # Create mask to ignore padding tokens (-100)
            mask = y_true_flat != -100
            y_pred_filtered = y_pred_flat[mask]
            y_true_filtered = y_true_flat[mask]
    
            # Calculate metrics only on valid positions
            accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
            f1_micro = f1_score(y_true_filtered, y_pred_filtered, average="micro", zero_division=0)
            f1_macro = f1_score(y_true_filtered, y_pred_filtered, average="macro", zero_division=0)
    
            # For per-class metrics (optional)
            f1_per_class = f1_score(y_true_filtered, y_pred_filtered, average=None, zero_division=0, labels=[0, 1, 2, 3])
    
            results[f"{partition}_accuracy"] = accuracy
            results[f"{partition}_f1_micro"] = f1_micro
            results[f"{partition}_f1_macro"] = f1_macro
            results[f"{partition}_f1_per_class"] = f1_per_class.tolist()  # Convert to list for saving
    #         y_pred = torch.cat([p[0] for p in predictions]).cpu().numpy()  # [N, 1024]
    #         y_true = torch.cat([p[1] for p in predictions]).cpu().numpy()  # [N, 1024]
    
    # # Flatten
    #         y_pred_flat = y_pred.flatten()  # [N*1024]
    #         y_true_flat = y_true.flatten()  # [N*1024]
    
    # # Filter out padding (-100)
    #         mask = y_true_flat != -100
    #         y_pred_filtered = y_pred_flat[mask]
    #         y_true_filtered = y_true_flat[mask]
    
    # # Calculate metrics only on real positions
    #         accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
    #         f1_micro = f1_score(y_true_filtered, y_pred_filtered, average="micro", zero_division=0)
    #         f1_macro = f1_score(y_true_filtered, y_pred_filtered, average="macro", zero_division=0, labels=[0,1,2,3])
    #         f1_per_class = f1_score(y_true_filtered, y_pred_filtered, average=None, zero_division=0, labels=[0,1,2,3])
    
    #         results[f"{partition}_accuracy"] = accuracy
    #         results[f"{partition}_f1_micro"] = f1_micro
    #         results[f"{partition}_f1_macro"] = f1_macro
    #         results[f"{partition}_f1_per_class"] = f1_per_class.tolist()

        elif cfg.task_name in ["ThermoStability","Thermostability"]:
            y_pred = torch.cat([p[0] for p in predictions]).cpu().numpy()
            y_true = torch.cat([p[1] for p in predictions]).cpu().numpy()
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            spearman_rho, _ = spearmanr(y_true, y_pred)
            results[f"{partition}_mse"] = mse
            results[f"{partition}_r2"] = r2
            results[f"{partition}_spearman_rho"] = spearman_rho

        else:  # multi_class
            y_pred = torch.cat([p[0] for p in predictions]).cpu().numpy()
            y_true = torch.cat([p[1] for p in predictions]).cpu().numpy()
            accuracy = accuracy_score(y_true, y_pred)
            f1_micro = f1_score(y_true, y_pred, average="micro")
            results[f"{partition}_accuracy"] = accuracy
            results[f"{partition}_f1_micro"] = f1_micro
            if cfg.task_name in ["TopEnzyme"]:
                if accuracy>0.80:
                    filename = f"/p/scratch/hai_oneprot/TopEnzyme_results_{cfg.seed}/TopEnzyme_{cfg.sweep.model_type}_{accuracy:.3f}_{partition}.txt"
                    data = np.column_stack((y_true, y_pred))
                    header = "y_true y_pred"
                    np.savetxt(filename, data, header=header, comments='', fmt='%s')




    return results

@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="saprot_mlp.yaml",
)
def main(cfg: DictConfig) -> None:
    #set_seed(42)
    #pl.seed_everything(cfg.seed, workers=True)
    #set_partial_seed(cfg.seed, deterministic_level="medium")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        print("CUDA is not available. Running on CPU.")

    # Generate all combinations of hyperparameters
    param_combinations = product(
        cfg.sweep.learning_rate,
        cfg.sweep.batch_size,
        cfg.sweep.max_epochs,
        cfg.sweep.hidden_dims,
        cfg.sweep.dropout_rate,
        cfg.sweep.use_batch_norm,
        cfg.sweep.use_layer_norm,
        cfg.sweep.activation,
        cfg.sweep.use_residual,
        cfg.sweep.task_name,
        cfg.sweep.model_type,
    )

    for (
        lr,
        batch_size,
        max_epochs,
        hidden_dims,
        dropout_rate,
        use_batch_norm,
        use_layer_norm,
        activation,
        use_residual,
        task_name,
        model_type,
    ) in param_combinations:
        # Update the configuration with the current hyperparameters
        cfg.model.learning_rate = lr
        cfg.model.batch_size = batch_size
        cfg.model.max_epochs = max_epochs
        cfg.model.hidden_dims = hidden_dims
        cfg.model.dropout_rate = dropout_rate
        cfg.model.use_batch_norm = use_batch_norm
        cfg.model.use_layer_norm = use_layer_norm
        cfg.model.activation = activation
        cfg.model.use_residual = use_residual
        cfg.task_name = task_name
        cfg.model_type = model_type

        data_module = EmbeddingDataModule(cfg)
        results = evaluate(cfg, data_module)

        # Save results to CSV using the utility function
        # Use different filename for merged binary tasks
        if task_name in ["ASD_merged_pocket_binary", "ASD_merged_pocket_sequence_binary","ASD_merged_pocket_binary_text","ASD_merged_pocket_sequence_binary_text","ASD_merged_pocket_binary_comp", "ASD_merged_pocket_sequence_binary_comp","ASD_merged_pocket_binary_text_comp","ASD_merged_pocket_sequence_binary_text_comp"]:
            # Temporarily modify task_name to append suffix for different file
            original_task_name = cfg.task_name
            cfg.task_name = f"{task_name}_balanced"
            save_results_to_csv(results, cfg)
            cfg.task_name = original_task_name
        else:
            save_results_to_csv(results, cfg)

        print(f"Results for {task_name}:")
        print(f"Learning rate: {lr}, Batch size: {batch_size}, Max epochs: {max_epochs}")
        print(f"Hidden dims: {hidden_dims}, Dropout rate: {dropout_rate}")
        print(f"Use batch norm: {use_batch_norm}, Use layer norm: {use_layer_norm}")
        print(f"Activation: {activation}, Use residual: {use_residual}")
        print(f"Model type: {model_type}")
        print(results)
        print("--------------------")


if __name__ == "__main__":
    main()
