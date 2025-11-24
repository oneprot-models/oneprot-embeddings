from torch_geometric.nn import inits, MessagePassing
from torch_geometric.nn import radius_graph
#sys.path.append('/p/project1/hai_oneprot/bazarova1/oneprot-panda/src')
#from .features import d_angle_emb, d_theta_phi_emb

from torch_scatter import scatter
from torch_sparse import matmul

import torch
from torch import nn
from torch.nn import Embedding
import torch.nn.functional as F

import numpy as np


num_aa_type = 40
num_side_chain_embs = 8
num_bb_embs = 6


def get_post_interaction_features(model, batch_data):
    """
    Extract features from ProNet right after the interaction blocks
    
    Args:
        model: The StructEncoder model
        batch_data: The input graph data
        
    Returns:
        Node features after interaction blocks and before final output layers
    """
    # Access the ProNet encoder
    struct_encoder = model.network['struct_graph']
    pronet = struct_encoder.encoder
    
    # Get all the input variables from batch_data
    z, pos, batch = torch.squeeze(batch_data.x.long()), batch_data.coords_ca, batch_data.batch
    # pos_n = batch_data.coords_n
    # pos_c = batch_data.coords_c
    # bb_embs = batch_data.bb_embs
    # side_chain_embs = batch_data.side_chain_embs
    device = z.device
    
    z = torch.squeeze(batch_data.x.long())
    pos = batch_data.coords_ca
    batch = batch_data.batch
    device = z.device
    
    # Debug info
    print(f"Debug - z shape: {z.shape}, min: {z.min().item()}, max: {z.max().item()}")
    
    # Create missing attributes if needed
    if not hasattr(batch_data, 'coords_n'):
        print("Creating missing coords_n")
        batch_data.coords_n = pos + torch.tensor([-1.0, 0.0, 0.0], device=device).unsqueeze(0)
    
    if not hasattr(batch_data, 'coords_c'):
        print("Creating missing coords_c")
        batch_data.coords_c = pos + torch.tensor([1.0, 0.0, 0.0], device=device).unsqueeze(0)
    
    if not hasattr(batch_data, 'bb_embs'):
        print("Creating missing bb_embs")
        batch_data.bb_embs = torch.zeros((z.size(0), num_bb_embs), device=device)
    
    if not hasattr(batch_data, 'side_chain_embs'):
        print("Creating missing side_chain_embs")
        batch_data.side_chain_embs = torch.zeros((z.size(0), num_side_chain_embs), device=device)
    
    pos_n = batch_data.coords_n
    pos_c = batch_data.coords_c
    bb_embs = batch_data.bb_embs
    side_chain_embs = batch_data.side_chain_embs
    
    # Check embedding layer dimensions
    emb_in_features = pronet.embedding.weight.shape[1] if hasattr(pronet.embedding, 'weight') else 40
    print(f"ProNet embedding expects input dimension: {emb_in_features}")
    
    # Process through initial embedding with dimension adaptation
    if pronet.level == 'aminoacid':
        x = pronet.embedding(z)
    elif pronet.level == 'backbone':
        # Create one-hot encoding
        one_hot = F.one_hot(z, num_classes=num_aa_type).float()
        
        # Calculate expected dimension vs actual dimension
        expected_dim = num_aa_type + bb_embs.shape[1]
        print(f"Expected feature dim: {expected_dim}, Embedding expects: {emb_in_features}")
        
        # Adapt dimensions if needed
        if expected_dim != emb_in_features:
            if expected_dim < emb_in_features:
                # Add padding
                padding_size = emb_in_features - expected_dim
                padding = torch.zeros((z.size(0), padding_size), device=device)
                x = torch.cat([one_hot, bb_embs, padding], dim=1)
                print(f"Added padding to match {emb_in_features} dimensions")
            else:
                # Truncate
                print(f"Warning: Feature dim {expected_dim} > embedding dim {emb_in_features}, truncating")
                combined = torch.cat([one_hot, bb_embs], dim=1)
                x = combined[:, :emb_in_features]
        else:
            x = torch.cat([one_hot, bb_embs], dim=1)
        
        print(f"Final input shape before embedding: {x.shape}")
        x = pronet.embedding(x)
    elif pronet.level == 'allatom':
        print("Using allatom level")
    # Create one-hot encoding
        one_hot = F.one_hot(z, num_classes=num_aa_type).float()
    
    # Calculate expected dimension vs actual dimension
        expected_dim = num_aa_type + bb_embs.shape[1] + side_chain_embs.shape[1]
        print(f"Expected feature dim: {expected_dim}, Embedding expects: {emb_in_features}")
    
    # Adapt dimensions if needed
        if expected_dim != emb_in_features:
            if expected_dim < emb_in_features:
            # Add padding
                padding_size = emb_in_features - expected_dim
                padding = torch.zeros((z.size(0), padding_size), device=device)
                x = torch.cat([one_hot, bb_embs, side_chain_embs, padding], dim=1)
                print(f"Added padding to match {emb_in_features} dimensions")
            else:
            # Truncate
                print(f"Warning: Feature dim {expected_dim} > embedding dim {emb_in_features}, truncating")
                combined = torch.cat([one_hot, bb_embs, side_chain_embs], dim=1)
                x = combined[:, :emb_in_features]
        else:
            x = torch.cat([one_hot, bb_embs, side_chain_embs], dim=1)
    
        print(f"Final input shape before embedding (allatom): {x.shape}")
        x = pronet.embedding(x)
    
    # Build the graph
    edge_index = radius_graph(pos, r=pronet.cutoff, batch=batch, max_num_neighbors=pronet.max_num_neighbors)
    pos_emb = pronet.pos_emb(edge_index, pronet.num_pos_emb)
    j, i = edge_index
    
    # Calculate all the features and angles
    # (This replicates the calculation from the ProNet forward method)
    dist = (pos[i] - pos[j]).norm(dim=1)
    num_nodes = len(z)
    
    # Calculate angles theta and phi
    refi0 = (i-1) % num_nodes
    refi1 = (i+1) % num_nodes
    
    a = ((pos[j] - pos[i]) * (pos[refi0] - pos[i])).sum(dim=-1)
    b = torch.cross(pos[j] - pos[i], pos[refi0] - pos[i]).norm(dim=-1)
    theta = torch.atan2(b, a)
    
    plane1 = torch.cross(pos[refi0] - pos[i], pos[refi1] - pos[i])
    plane2 = torch.cross(pos[refi0] - pos[i], pos[j] - pos[i])
    a = (plane1 * plane2).sum(dim=-1)
    b = (torch.cross(plane1, plane2) * (pos[refi0] - pos[i])).sum(dim=-1) / ((pos[refi0] - pos[i]).norm(dim=-1))
    phi = torch.atan2(b, a)
    
    # Calculate feature0
    feature0 = pronet.feature0(dist, theta, phi)
    
    # Calculate feature1 based on level
    if pronet.level == 'backbone' or pronet.level == 'allatom':
        # Calculate Euler angles
        Or1_x = pos_n[i] - pos[i]
        Or1_z = torch.cross(Or1_x, torch.cross(Or1_x, pos_c[i] - pos[i]))
        Or1_z_length = Or1_z.norm(dim=1) + 1e-7
        
        Or2_x = pos_n[j] - pos[j]
        Or2_z = torch.cross(Or2_x, torch.cross(Or2_x, pos_c[j] - pos[j]))
        Or2_z_length = Or2_z.norm(dim=1) + 1e-7
        
        Or1_Or2_N = torch.cross(Or1_z, Or2_z)
        
        angle1 = torch.atan2((torch.cross(Or1_x, Or1_Or2_N) * Or1_z).sum(dim=-1)/Or1_z_length, (Or1_x * Or1_Or2_N).sum(dim=-1))
        angle2 = torch.atan2(torch.cross(Or1_z, Or2_z).norm(dim=-1), (Or1_z * Or2_z).sum(dim=-1))
        angle3 = torch.atan2((torch.cross(Or1_Or2_N, Or2_x) * Or2_z).sum(dim=-1)/Or2_z_length, (Or1_Or2_N * Or2_x).sum(dim=-1))
        
        if pronet.euler_noise:
            euler_noise = torch.clip(torch.empty(3,len(angle1)).to(device).normal_(mean=0.0, std=0.025), min=-0.1, max=0.1)
            angle1 += euler_noise[0]
            angle2 += euler_noise[1]
            angle3 += euler_noise[2]
        
        feature1 = torch.cat((pronet.feature1(dist, angle1), pronet.feature1(dist, angle2), pronet.feature1(dist, angle3)), 1)
    else:
        # Aminoacid level
        refi = (i-1) % num_nodes
        
        refj0 = (j-1) % num_nodes
        refj = (j-1) % num_nodes
        refj1 = (j+1) % num_nodes
        
        mask = refi0 == j
        refi[mask] = refi1[mask]
        mask = refj0 == i
        refj[mask] = refj1[mask]
        
        plane1 = torch.cross(pos[j] - pos[i], pos[refi] - pos[i])
        plane2 = torch.cross(pos[j] - pos[i], pos[refj] - pos[j])
        a = (plane1 * plane2).sum(dim=-1)
        b = (torch.cross(plane1, plane2) * (pos[j] - pos[i])).sum(dim=-1) / dist
        tau = torch.atan2(b, a)
        
        feature1 = pronet.feature1(dist, tau)
    
    # Process through all interaction blocks
    for interaction_block in pronet.interaction_blocks:
        if pronet.data_augment_eachlayer:
            # add gaussian noise to features
            gaussian_noise = torch.clip(torch.empty(x.shape).to(device).normal_(mean=0.0, std=0.025), min=-0.1, max=0.1)
            x += gaussian_noise
        x = interaction_block(x, feature0, feature1, pos_emb, edge_index, batch)
    
    # At this point, x contains node features after all interaction blocks
    # This is what we want - the output right after the interaction blocks
    
    # Instead of proceeding with scatter and final layers, return the output 
    # Create a new Data object with the node features and original graph structure
    post_interaction_data = Data(
        x=x,
        edge_index=edge_index,
        coords_ca=pos,
        batch=batch
    )
    
    return post_interaction_data