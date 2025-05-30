import torch
import torch.nn as nn
from torch import TensorType
import sys
import os
import argparse



# Add the external mdgen directory to path to import LatentMDGenModel
sys.path.append("/p/project1/hai_oneprot/bazarova1/oneprot-panda/external/mdgen")

#from mdgen.parsing import parse_train_args
from mdgen.model.latent_model import LatentMDGenModel
from src.models.components.base_encoder import BaseEncoder

class MDGenArgs:
    """Manual replacement for MDGen's parse_train_args"""
    def __init__(self):
        # Core model parameters
        self.hidden_size = 768
        self.num_layers = 6
        self.num_heads = 12
        self.dropout = 0.1
        self.mha_heads = 16

        #self.num_frames =   # Add this missing attribute
        self.frame_interval = None
        self.cond_interval = None
        self.crop = 1024
        self.atlas = True
        self.data_dir = "/p/scratch/hai_profound/data/mdCATH_data"
        self.copy_frames = False
        self.overfit = False
        self.overfit_peptide = None
        self.overfit_frame = False
        self.suffix = "_i40"
        
        # Other required parameters for LatentMDGenModel
        self.tps_condition = False
        self.no_torsion = False
        self.no_frames = False
        self.supervise_no_torsions = False
        self.hyena = False
        self.no_rope = False
        self.embed_dim = 384
        self.ipa_heads = 8
        self.ipa_head_dim = 16
        self.ipa_qk = None
        self.ipa_v = None
        self.time_multiplier = 1.0
        self.abs_pos_emb = False
        self.abs_time_emb = False
        self.path_type = "Linear"  # Options: Linear, GVP, VP
        self.prediction = "velocity"  # Options: velocity, score, noise
        self.sampling_method = "dopri5"  # Options: dopri5, euler
        self.alpha_max = 1.0
        self.discrete_loss_weight = 1.0
        self.dirichlet_flow_temp = 1.0
        self.allow_nan_cfactor = False
        self.scale_factor = 1.0
        
        # Design-related parameters
        self.design = False
        self.design_from_traj = False
        self.sim_condition = False
        self.inpainting = False
        self.dynamic_mpnn = False
        self.mpnn = False
        self.prepend_ipa= False
        self.interleave_ipa= False
        self.grad_checkpointing = False  # Add this if needed for LatentMDGenModel

class TrajectoryEncoder(BaseEncoder):
    def __init__(
        self,
        output_dim: int,
        model_path: str = '/p/scratch/profound/bazarova1/forward_sim.ckpt',
        pooling_type: str = "mean",
        proj_type: str = None,
        use_logit_scale: bool = False,
        learnable_logit_scale: bool = False,
        pretrained: bool = False,
        frozen: bool = True,
        hidden_size: int = 384,
        num_layers: int = 6,
        num_heads: int = 12,
        dropout: float = 0.1,
        num_frames: int = 39,
        suffix: str = "_i40",
    ):
        """
        Encoder for MD trajectories using LatentMDGenModel
        
        Args:
            output_dim: Dimensionality of the output embedding
            model_path: Path to the pretrained LatentMDGenModel checkpoint
            pooling_type: Type of pooling to use ('mean', 'cls', etc.)
            proj_type: Projection head type
            use_logit_scale: Whether to use logit scaling
            learnable_logit_scale: Whether logit scale should be learnable
            pretrained: Whether to load a pretrained model
            frozen: Whether to freeze the transformer parameters
            hidden_size: Hidden size for the model if not pretrained
            num_layers: Number of layers for the model if not pretrained
            num_heads: Number of attention heads if not pretrained
            dropout: Dropout rate if not pretrained
        """
        # For now, we'll use a fixed hidden_size
        self.hidden_size = hidden_size
        
        super().__init__(
            d_model=self.hidden_size,
            output_dim=output_dim,
            proj_type=proj_type,
            use_logit_scale=use_logit_scale,
            learnable_logit_scale=learnable_logit_scale,
            pooling_type=pooling_type
        )
        
        # Load the MDGen model
        if pretrained and model_path is not None:
            #self.transformer = LatentMDGenModel.load_from_checkpoint('/p/scratch/profound/bazarova1/forward_sim.ckpt')
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                    # Extract just the model part (remove 'model.' prefix if needed)
                state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() 
                                 if k.startswith('model.')}
                    # Or if stored differently:
                    # state_dict = checkpoint['state_dict']
            else:
                    # Raw state dict
                state_dict = checkpoint
            args=MDGenArgs()
            args.hidden_size = hidden_size
            args.num_layers = num_layers
            args.num_heads = num_heads
            args.dropout = dropout
            args.num_frames = num_frames
            args.suffix = suffix
            # Initialize a default model with the required args
            self.transformer = LatentMDGenModel(args=args, latent_dim=21)
            self.transformer.load_state_dict(state_dict, strict=False)

            print("loaded pretrained LatentMDGenModel from checkpoint")
        else:
            print("initializing LatentMDGenModel with custom settings")
            # Create default args for LatentMDGenModel
            #args = parse_train_args()
            # Get the default arguments
            
            # Override defaults with our custom settings
            args=MDGenArgs()
            args.hidden_size = hidden_size
            args.num_layers = num_layers
            args.num_heads = num_heads
            args.dropout = dropout
            args.num_frames = num_frames
            
            # Initialize a default model with the required args
            self.transformer = LatentMDGenModel(args=args, latent_dim=21)
        
 
            
            # Add any other required args here
            # Initialize a default model with the required args        
        # Update hidden size in case it's different from what we expected
        if hasattr(self.transformer, 'hidden_size'):
            self.hidden_size = self.transformer.hidden_size
            
            # Reinitialize components that depend on hidden_size if it changed
            if self.hidden_size != self.d_model:
                self.d_model = self.hidden_size
                self._init_pooling(pooling_type)
                self._init_projection(proj_type)
                self._init_norm()
        
        if frozen:
            for param in self.transformer.parameters():
                param.requires_grad = False
    
    def forward(self, x, t=0, mask=None, start_frames=None, end_frames=None, 
            x_cond=None, x_cond_mask=None, aatype=None):
        """
        Forward pass for trajectory data
        
        Args:
            x: Input trajectory data appropriate for the LatentMDGenModel
                This might be torsions or other trajectory representation
            
        Returns:
            Normalized projected embeddings
        """
        # Generate attention mask - this might need customization based on your data
        #print(x.shape, " x!!!!!!!!!")
        if mask is None:
            if len(x.shape) >= 3:  # [batch, time, length, ...]
                mask = torch.ones(x.shape[0], x.shape[2], device=x.device)
            else:
                mask = torch.ones(x.shape[0], x.shape[1], device=x.device)
    
    # Ensure t is a tensor
        if isinstance(t, (int, float)):
            t = torch.tensor([t] * x.shape[0], device=x.device)
    
    # Forward pass through transformer using the same arguments
        outputs = self.transformer(
            x=x, 
            t=t,
            mask=mask,
            start_frames=start_frames,
            end_frames=end_frames,
            x_cond=x_cond,
            x_cond_mask=x_cond_mask,
            aatype=aatype
            )
        
        # Get the last hidden state or appropriate output from the model
        if isinstance(outputs, torch.Tensor):
            last_hidden_state = outputs
        elif hasattr(outputs, 'last_hidden_state'):
            last_hidden_state = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            last_hidden_state = outputs.hidden_states[-1]
        else:
            raise ValueError(f"Cannot extract hidden states from model outputs: {type(outputs)}")
        
        # Apply pooling
        # print(outputs.shape," shape outputs")
        # print(mask.shape, " shape mask")
        # print(last_hidden_state, " shape last_hidden_state")
        # print(torch.isnan(last_hidden_state).sum(), " number of NaNs in tensor")
        #print(outputs, " shape outputs")
        if len(last_hidden_state.shape) == 4:  # [batch, time, length, channels]
            # First pool over time dimension
            pooled_time = last_hidden_state.mean(dim=1)  # [batch, length, channels]
        # Then pool over length dimension with mask

            if len(mask.shape) == 3:  # [batch, time, length]
        # Convert mask from [batch, time, length] to [batch, length]
        # by taking the min along time dim (only consider a position valid if it's valid at all timesteps)
                mask_2d = mask.min(dim=1)[0]  # [batch, length]
                #print(f"Reshaped mask from {mask.shape} to {mask_2d.shape}")
            else:
                mask_2d = mask
            pooled_output = self.pooling(pooled_time, mask_2d)
        elif len(last_hidden_state.shape) == 3:  # [batch, length, channels]
        # Direct pooling with mask
            pooled_output = self.pooling(last_hidden_state, mask)
        else:
            raise ValueError(f"Unexpected shape for last_hidden_state: {last_hidden_state.shape}")
 
        #print(pooled_output.shape, " shape pooled_output")
        # Apply projection and normalization
    
        projected = self.proj(pooled_output)
        return self.norm(projected)