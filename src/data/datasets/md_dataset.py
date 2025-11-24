

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, EsmTokenizer
import pandas as pd
from typing import List, Tuple
import h5py
import sys
sys.path.append("external")

from mdgen.mdgen.rigid_utils import Rigid
from mdgen.mdgen.residue_constants import restype_order, restype_atom37_mask
import numpy as np
import pandas as pd
from mdgen.mdgen.geometry import atom37_to_torsions, atom14_to_atom37, atom14_to_frames
from mdgen.mdgen.wrapper import NewMDGenWrapper
#from mdgen.mdgen.parsing import parse_train_args

# Add this class definition after the imports:
class MDGenArgsData:
    """Manual replacement for MDGen's parse_train_args"""
    def __init__(self):
        # Core dataset parameters
        self.atlas = True
        self.crop = 1024
        self.num_frames = 39
        self.frame_interval = None
        self.data_dir = "/p/scratch/hai_profound/data/mdCATH_data"
        self.no_frames = False
        self.copy_frames = False
        self.overfit = False
        self.overfit_peptide = None
        self.overfit_frame = False
        self.suffix = "_i40"
        
        # Other required parameters
        self.no_torsion = False
        self.no_design_torsion = False
        self.supervise_no_torsions = False
        self.supervise_all_torsions = False
        self.no_offsets = False
        self.hyena = False
        self.no_rope = False
        self.dropout = 0.1
        self.scale_factor = 1.0
        self.interleave_ipa = False
        self.prepend_ipa = False
        self.oracle = False
        self.num_layers = 6
        self.embed_dim = 768
        self.mha_heads = 12
        self.ipa_heads = 8
        self.ipa_head_dim = 16
        self.ipa_qk = None
        self.ipa_v = None
        self.time_multiplier = 1.0
        self.abs_pos_emb = False
        self.abs_time_emb = False
        self.path_type = "Linear"
        self.prediction = "velocity"
        self.sampling_method = "dopri5"
        self.alpha_max = 1.0
        self.discrete_loss_weight = 1.0
        self.dirichlet_flow_temp = 1.0
        self.allow_nan_cfactor = False
        self.tps_condition = False
        self.design = False
        self.design_from_traj = False
        self.sim_condition = False
        self.inpainting = False
        self.dynamic_mpnn = False
        self.mpnn = False
        self.cond_interval = None

class MDDataset(Dataset):
    def __init__(self, split: str, data_dir: str = '/p/scratch/hai_profound/data/mdCATH_data',
                 data_dir_csv: str = '/p/project1/hai_oneprot/bazarova1/mdgen/splits/mdCATH.csv',   
                 max_length: int = 1024,
                 seq_tokenizer: str = "facebook/esm2_t33_650M_UR50D", num_frames: int = 39, 
                 frame_interval: int = None, crop: int = 1024, overfit: bool = False,
                 overfit_peptide: str = None, overfit_frame: bool = False, 
                 copy_frames: bool = False, no_frames: bool = False, atlas: bool = True, 
                 suffix: str = "_i1", repeat: int = 1):

        super().__init__()
        self.df = pd.read_csv(f'{data_dir_csv}_{split}.csv', index_col='name')
        #self.df = pd.read_csv(f'/p/project1/hai_oneprot/bazarova1/mdgen/splits/atlas_train.csv', index_col='name')
        # if split=='train':
        #     self.df=self.df.iloc[:-1000]
        # else:
        #     self.df=self.df.iloc[-1000:]
        
        self.data_dir_csv = data_dir_csv
        self.data_dir = data_dir
        self.max_length = max_length
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.crop = crop
        self.overfit = overfit
        self.overfit_peptide = overfit_peptide
        self.overfit_frame = overfit_frame
        self.copy_frames = copy_frames
        self.no_frames = no_frames
        self.atlas = atlas
        self.suffix = suffix
        self.repeat = repeat
        self.seq_tokenizer = AutoTokenizer.from_pretrained(seq_tokenizer)
        #args=parse_train_args()

        args= MDGenArgsData()
        args.atlas = atlas  
        args.crop = crop
        args.frame_interval = frame_interval
        args.data_dir = data_dir
        args.no_frames = no_frames
        args.copy_frames = copy_frames
        args.overfit = overfit
        args.overfit_peptide = overfit_peptide
        args.overfit_frame = overfit_frame
        args.suffix = suffix

        self.mdgen_wrapper = NewMDGenWrapper(args)
        self.split = split

    
    def __len__(self):
        # if self.args.overfit_peptide:
        #     return 1000
        
        return self.repeat * len(self.df)
        


    def __getitem__(self, idx):
        return self.df.index[idx]

    def collate_fn(self,name):
#        idx = idx % len(self.df)
#        if self.args.overfit:
#            idx = 0

#        if self.args.overfit_peptide is None:
        #name = self.df.index[idx]
        #sequences = self.df.seqres[name]
        sequences = [str(self.df.seqres[n]) for n in name]
        # print(sequences," sequences", flush=True)
        # print("Sequences type:", type(sequences))
        sequence_input = self.seq_tokenizer(sequences, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt").input_ids  
#        else:
#            name = self.args.overfit_peptide
#            seqres = name

        if self.atlas:
            #i=1
            if 'combined' in self.data_dir_csv:
                replicas = [self.df.at[n, 'replicas'] for n in name]
                r_nums = []
                for replica_str in replicas:
                    #print(self.df.name[n], replica_str, "replica_str", flush=True)
                    if pd.notna(replica_str):                        
                # Split the string by comma and convert to list of integers
                        replica_options = [int(r) for r in str(replica_str).split(',')]
                # Choose a random replica from the available options
                        r_nums.append(np.random.choice(replica_options))
                    else:
                        r_num = np.random.randint(1, 4)

                        r_nums.append(r_num)
                full_name = [f"{n}_R{r}" for n, r in zip(name, r_nums)]
            else:
                #print(len(name), "name length", flush=True)
                full_name = []
                for n in name:
                    r_num = np.random.randint(1, 4)
 
                    full_name.append(f"{n}_R{r_num}")

        else:
            full_name = [n for n in name]


        trajectory_data_list=[]

        if type(full_name) is not list:
            #print(f"full_name: {full_name} not a list",flush=True)
            if len(full_name.split('_')[0])>4 and len(n.split('_')[0])<8:
                self.data_dir='/p/scratch/hai_profound/data/mdCATH_data'
                self.suffix='_i1'
            elif len(full_name.split('_')[0])>8:
                self.data_dir='/p/data1/profound_data/GPCRmd_processed'
                self.suffix='_i5'
            else:
                self.data_dir='/p/scratch/hai_profound/data/atlas_data'
                self.suffix='_i100'
            arr = np.lib.format.open_memmap(f'{self.data_dir}/{full_name}{self.suffix}.npy', 'r')
        else:
            #print(f"full_name: {full_name} is a list",flush=True)
            for i,n in enumerate(full_name):
                if len(n.split('_')[0])>4 and len(n.split('_')[0])<8:
                    #print(i,n,n.split('_')[0], "n name!!!!!!!!", flush=True)
                    self.data_dir='/p/scratch/hai_profound/data/mdCATH_data'
                    self.suffix='_i1'
                elif len(n.split('_')[0])>8:
                    #print(i,n,n.split('_')[0], "n name!!!!!!!!", flush=True)
                    self.data_dir='/p/data1/profound_data/GPCRmd_processed'
                    self.suffix='_i5'
                else:
                    self.data_dir='/p/scratch/hai_profound/data/atlas_data'
                    self.suffix='_i100'
                #print(n, "n name!!!!!!!!", flush=True)
                arr = np.lib.format.open_memmap(f'{self.data_dir}/{n}{self.suffix}.npy', 'r')
                #print(arr.shape, "arr shape", flush=True)
                #print(arr.shape, "arr shape", flush=True)
                if self.frame_interval:
                    arr = arr[::self.frame_interval]

                if arr.shape[0] - self.num_frames<=0:
                    print(n, arr.shape,arr.shape[0] - self.num_frames, "name and arr shape!!!!!!!!!!", flush=True)
                frame_start = np.random.choice(np.arange(arr.shape[0] - self.num_frames))
                if self.overfit_frame:
                    frame_start = 0
                end = frame_start + self.num_frames
                # arr = np.copy(arr[frame_start:end]) * 10 # convert to angstroms
                arr = np.copy(arr[frame_start:end]).astype(np.float32) # / 10.0 # convert to nm
                if self.copy_frames:
                    arr[1:] = arr[0]

                # arr should be in ANGSTROMS
                frames = atom14_to_frames(torch.from_numpy(arr))
                seqres = np.array([restype_order[c] for c in sequences[i]])
                aatype = torch.from_numpy(seqres)[None].expand(self.num_frames, -1)
                # print(f"arr shape: {arr.shape}, {n}")
                # print(f"aatype shape: {aatype.shape}")
                # print(f"type of arr: {type(arr)}")
                # print(f"type of aatype: {type(aatype)}")
                if arr.shape[1] !=len(seqres):
                    print(n, " arr shape does not match seqres length", arr.shape, len(seqres), flush=True)
        
                atom37 = torch.from_numpy(atom14_to_atom37(arr, aatype)).float()

                L = frames.shape[1]
                mask = np.ones(L, dtype=np.float32)
        
                if self.no_frames:
                    return {
                        'name': full_name,
                        'frame_start': frame_start,
                        'atom37': atom37,
                        'seqres': seqres,
                        'mask': restype_atom37_mask[seqres], # (L,)
                    }
                torsions, torsion_mask = atom37_to_torsions(atom37, aatype)
        
                torsion_mask = torsion_mask[0]
        
                if self.atlas:
                    if L > self.crop:
                        start = np.random.randint(0, L - self.crop + 1)
                        torsions = torsions[:,start:start+self.crop]
                        frames = frames[:,start:start+self.crop]
                        seqres = seqres[start:start+self.crop]
                        mask = mask[start:start+self.crop]
                        torsion_mask = torsion_mask[start:start+self.crop]
                
            
                    elif L < self.crop:
                        pad = self.crop - L
                        frames = Rigid.cat([
                            frames, 
                            Rigid.identity((self.num_frames, pad), requires_grad=False, fmt='rot_mat')
                        ], 1)
                        mask = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])
                        seqres = np.concatenate([seqres, np.zeros(pad, dtype=int)])
                        torsions = torch.cat([torsions, torch.zeros((torsions.shape[0], pad, 7, 2), dtype=torch.float32)], 1)
                        torsion_mask = torch.cat([torsion_mask, torch.zeros((pad, 7), dtype=torch.float32)])

        

                trajectory_data={
                    'name': full_name,
                    'frame_start': frame_start,
                    'torsions': torsions.unsqueeze(0),
                    'torsion_mask': torsion_mask.unsqueeze(0),
                    'trans': frames._trans.unsqueeze(0),
                    'rots': frames._rots._rot_mats.unsqueeze(0),
                    'seqres': torch.from_numpy(seqres).unsqueeze(0),
                    'mask': torch.from_numpy(mask).unsqueeze(0), # (L,)
                    }
        
                trajectory_data_list.append(trajectory_data)

            combined_traj = {
                'name': full_name,
                'frame_start': [td['frame_start'] for td in trajectory_data_list],
                'torsions': torch.cat([td['torsions'] for td in trajectory_data_list]),
                'torsion_mask': torch.cat([td['torsion_mask'] for td in trajectory_data_list]),
                'trans': torch.cat([td['trans'] for td in trajectory_data_list]),
                'rots': torch.cat([td['rots'] for td in trajectory_data_list]),
                'seqres': torch.cat([td['seqres'] for td in trajectory_data_list]),
                'mask': torch.cat([td['mask'] for td in trajectory_data_list]),
            }


            mdgen_batch=self.mdgen_wrapper.prep_batch(combined_traj)

        modality='md'

        return sequence_input, mdgen_batch, modality, sequences
