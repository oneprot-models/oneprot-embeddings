import hydra
from pathlib import Path
import os
import sys
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, LightningDataModule
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import distributed
import h5py
from torch_geometric.data import Batch
import pandas as pd
import random
from datetime import datetime



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.utils.struct_graph_utils import protein_to_graph
from src.data.datasets.text_dataset_instr import TextDatasetInstr


from downstream.pandagpt.galactica_utils import StoppingCriteriaSub, define_prompt_wrap, encode_modality, process_batch_instance, prepare_generation_embedding


# Add these imports at the top
import os
from datetime import datetime

@hydra.main(version_base="1.3", config_path="../configs", config_name="panda_gpt.yaml")
def save_checkpoint(model, modality_proj, optimizer, epoch, loss, acc, cfg, checkpoint_dir):
    """
    Save a checkpoint of the model, projection layer, and training state.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_{timestamp}.pt')
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'modality_proj_state_dict': modality_proj.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': acc,
        'config': OmegaConf.to_container(cfg, resolve=True)
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, modality_proj, optimizer, device):
    """
    Load a checkpoint and restore the training state.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.module.load_state_dict(checkpoint['model_state_dict'])
    
    # Load projection layer state
    modality_proj.module.load_state_dict(checkpoint['modality_proj_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy'], checkpoint['config']

@hydra.main(version_base="1.3", config_path="../configs", config_name="panda_gpt.yaml")
def main(cfg):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # print(f"SLURM_PROCID: {os.environ.get('SLURM_PROCID')}")
    # print(f"SLURM_LOCALID: {os.environ.get('SLURM_LOCALID')}")
    # print(f"SLURM_NTASKS: {os.environ.get('SLURM_NTASKS')}")
    rank = int(os.environ.get('SLURM_PROCID', '0'))
    world_size = int(os.environ.get('SLURM_NTASKS', '1'))
    local_rank = int(os.environ.get('SLURM_LOCALID', '0'))
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '12345')

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # if device == 0:
    #     print("entered main!!!!!!!!!!!!!!!!")
    #     print(f"{device} device!!!!!!!!!!!!!!!")
        # print(f"{world_size} world_size!!!!!!!!!!!!!!!")
        # print(f"{master_addr} {master_port} master_addr!!!!!!!!!!!!!!!")

    # if torch.cuda.is_available():
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     torch.backends.cudnn.benchmark = True
    #     torch.backends.cudnn.deterministic = False

    # # Initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{master_addr}:{master_port}',
        rank=rank,
        world_size=world_size
    )
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        #distributed.init_distributed_mode(12354)

    # Load Galactica model
    model_path = '/p/scratch/hai_oneprot/huggingface/models/facebook/galactica-6.7b'
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=cfg.lora_r, 
        lora_alpha=cfg.lora_alpha, 
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules
    )

    checkpoint = torch.load('/p/scratch/hai_oneprot/bazarova1/checkpoints_pa/2025-01-30_220904/checkpoint_epoch1_valloss2.7549.pt', map_location=device)

    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model = get_peft_model(model, peft_config)

    
    new_state_dict = {}

    # print(checkpoint['model_state_dict'].items(),"checkpoint!!!!!!!!!!")
    # print(model.state_dict().items(),"model!!!!!!!!!!")

    for k, v in checkpoint['model_state_dict'].items():
            # Handle cases where keys might need adjustment
        if k.startswith('module.'):
                # Remove extra 'model.' if present
            new_k = k.replace('module.', '')
            new_state_dict[new_k] = v
        # if k.startswith('module.base_model.model.lm_head.weight'):
        #     new_k=k.replace('module.base_model.model.lm_head.weight','base_model.model.lm_head.weight')
        #     new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v


#    print(new_state_dict.items(),"new_state_dict!!!!!!!!!!")

#    model.load_state_dict(new_state_dict)
    model.print_trainable_parameters()
    model = DDP(model, device_ids=[local_rank],bucket_cap_mb=25)

    # Load OneProt and freeze all parameters
    model_config_path = '/p/project1/hai_oneprot/bazarova1/oneprot-refined/logs/train/runs/2024-11-05_16-37-44/config.yaml'
    #model_config_path = '/p/project1/hai_oneprot/bazarova1/oneprot-refined/logs/train/runs/2024-10-16_22-32-28/config.yaml'

    model_cfg = OmegaConf.load(model_config_path)
    oneprot = hydra.utils.instantiate(model_cfg.model)
    oneprot.load_state_dict(torch.load('/p/scratch/hai_oneprot/checkpoints_refined_111024/2024-11-05_19-20-45/epoch_043_28400.ckpt')["state_dict"])
    #oneprot.load_state_dict(torch.load('/p/scratch/hai_oneprot/checkpoints_refined_111024/2024-10-16_22-32-28/best-v16.ckpt')["state_dict"])
    one_prot_output_dim = oneprot.network[cfg.modality].output_dim
    modality_encoder = oneprot.network[cfg.modality].to(device)
    modality_encoder.eval()
    for param in modality_encoder.parameters():
        param.requires_grad = False

    # Prepare tokenizer
    tokenizer_local = AutoTokenizer.from_pretrained(model_path)
    tokenizer_local.bos_token_id, tokenizer_local.eos_token_id = 1, 2
    tokenizer_local.pad_token = tokenizer_local.eos_token
    tokenizer_local.padding_side = "right"

    # Create learnable projection
    modality_proj = nn.Linear(
                one_prot_output_dim, model.module.config.hidden_size
            ).to(device)

    new_state_dict = {}

    for k, v in checkpoint['proj_state_dict'].items():
            # Handle cases where keys might need adjustment
        if k.startswith('module.'):
                # Remove extra 'model.' if present
            new_k = k.replace('module.', '')
            new_state_dict[new_k] = v
        # if k.startswith('module.base_model.model.lm_head.weight'):
        #     new_k=k.replace('module.base_model.model.lm_head.weight','base_model.model.lm_head.weight')
        #     new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    
#    modality_proj.load_state_dict(new_state_dict)

    modality_proj = DDP(modality_proj, device_ids=[local_rank], bucket_cap_mb=25)
    # if device == 0:
    #     print("loaded modality projection with dims: ", one_prot_output_dim, model.module.config.hidden_size)

    # Load training data for finetuning
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    if rank == 0:
        print("before dataloader")
    # modality_dl = DataLoader(
    #                     dataset=datamodule.datasets[f"{cfg.modality_data_name}_train"],
    #                     batch_size=32,
    #                     num_workers=datamodule.num_workers,
    #                     pin_memory=datamodule.pin_memory,
    #                     collate_fn=datamodule.datasets[f"{cfg.modality_data_name}_train"].text_collate_fn,
    #                     drop_last=True,
    #                     shuffle=False,
    #                     sampler=DistributedSampler(datamodule.datasets[f"{cfg.modality_data_name}_train"], shuffle=True, rank=rank, num_replicas=world_size)
    #                 )

    kwargs = {**cfg.data.modalities.text.dataset, 'split': 'train'}

    instruction_dl = DataLoader(
                dataset=TextDatasetInstr(**kwargs),
                        batch_size=32,
                        num_workers=datamodule.num_workers,
                        pin_memory=datamodule.pin_memory,
                        collate_fn=TextDatasetInstr(**kwargs).text_collate_fn,
                        drop_last=True,
                        shuffle=False,
                        sampler=DistributedSampler(TextDatasetInstr(**kwargs), shuffle=True, rank=rank, num_replicas=world_size)
                    )
            

    # After training dataloader creation, add validation dataloader
    val_dl = DataLoader(
        dataset=datamodule.datasets[f"{cfg.modality_data_name}_val"],
        batch_size=32,
        num_workers=datamodule.num_workers,
        pin_memory=datamodule.pin_memory,
        collate_fn=datamodule.datasets[f"{cfg.modality_data_name}_val"].text_collate_fn,
        drop_last=False,
        shuffle=False,
        sampler=DistributedSampler(datamodule.datasets[f"{cfg.modality_data_name}_val"], 
                              shuffle=False, 
                              rank=rank, 
                              num_replicas=world_size)
        )

    # val_dl_structure = DataLoader(
    #     dataset=datamodule.datasets["struct_graph_val"],
    #     batch_size=32,
    #     num_workers=datamodule.num_workers,
    #     pin_memory=datamodule.pin_memory,
    #     collate_fn=datamodule.datasets["struct_graph_val"].text_collate_fn,
    #     drop_last=False,
    #     shuffle=False,
    #     sampler=DistributedSampler(datamodule.datasets["struct_graph_val"], 
    #                           shuffle=False, 
    #                           rank=rank, 
    #                           num_replicas=world_size)
    #     )

    def validate(model, val_dl, modality_encoder, modality_proj, tokenizer_local, device, cfg, rank, world_size):
        model.eval()
        val_losses = []
        val_accuracies = []
    
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dl):
                modality_input, text_input, _ = batch
                modality_input = modality_input.to(device)
                modality_embs = encode_modality(modality_input, modality_encoder, modality_proj).to(device)
            
                text_input = [[{"from": "human", "value": cfg.dummy_prompt}, {"from": "gpt", "value": t}] for t in text_input]
                input_ids, target_ids, attention_mask = process_batch_instance(tokenizer_local, text_input, cfg.max_tgt_len)
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                attention_mask = attention_mask.to(device)
                inputs_embeds, targets, attention_mask = define_prompt_wrap(modality_embs, input_ids, target_ids, attention_mask, model, tokenizer_local)
            
                outputs = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            
                local_loss = outputs.loss
            
                chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]
                labels = targets[:, 2:]
                gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)
                valid_mask = (labels != -100).reshape(-1)
                valid_tokens = gen_acc & valid_mask
                local_acc = valid_tokens.sum().item() / valid_mask.sum().item()
            
                # Gather metrics
                gathered_losses = [torch.zeros_like(local_loss) for _ in range(world_size)]
                gathered_accs = [torch.zeros_like(torch.tensor(local_acc, device=device)) for _ in range(world_size)]
            
                dist.all_gather(gathered_losses, local_loss)
                dist.all_gather(gathered_accs, torch.tensor(local_acc, device=device))
            
                val_losses.append(torch.mean(torch.stack(gathered_losses)).item())
                val_accuracies.append(torch.mean(torch.stack(gathered_accs)).item())
    
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_acc = sum(val_accuracies) / len(val_accuracies)
        return avg_val_loss, avg_val_acc

    # In the training loop, after training epoch


    

    if rank == 0:
        print("after dataloader")
        print("Modality dataloader length: ", len(instruction_dl))
    
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': modality_proj.parameters()}
        ], lr=cfg.learning_rate)

    num_warmup_steps = int(0.1 * cfg.num_train_epochs * len(train_dataloader))  # 10% of total steps
    
    scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=cfg.num_train_epochs * len(train_dataloader)
    )
    
    #optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
#    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    for epoch in range(cfg.num_epochs):
        instruction_dl.sampler.set_epoch(epoch)  # Set epoch for proper shuffling
        batch_losses = []
        batch_accuracies = []
        #for batch_idx, batch in enumerate(modality_dl):
        for batch_idx, batch in enumerate(instruction_dl):
          
            modality_input, text_prompt, text_output = batch
            modality_input = modality_input.to(device)
            modality_embs = encode_modality(modality_input, modality_encoder, modality_proj).to(device)
            # print(f"[Rank {rank}] Embeddings shape: {modality_embs.shape}")
            

            # print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}")
            # print(f"Text input samples on rank {rank}:", text_input[:2])  # Show first 2 samples
            
            # # Verify tensor device placement
            # print(f"Rank {rank}, Device: {modality_input.device}")

            #dist.barrier()
            # Force sync point to keep output readable

            
            text_input = [[{"from": "human", "value": t0}, {"from": "gpt", "value": t}] for t0,t in zip(text_prompt,text_output)]
            input_ids, target_ids, attention_mask = process_batch_instance(tokenizer_local, text_input, cfg.max_tgt_len)
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            attention_mask = attention_mask.to(device)
            inputs_embeds, targets, attention_mask = define_prompt_wrap(modality_embs, input_ids, target_ids, attention_mask, model, tokenizer_local)
        
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        
            local_loss = outputs.loss
            optimizer.zero_grad()
            local_loss.backward()
            optimizer.step()
        
            chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]    # [B, S-1]
            labels = targets[:, 2:]
            gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)    # [B*S]
            valid_mask = (labels != -100).reshape(-1)
            valid_tokens = gen_acc & valid_mask    # [B*S]
            local_acc = valid_tokens.sum().item() / valid_mask.sum().item()
                
            # Gather metrics from all GPUs
            gathered_losses = [torch.zeros_like(local_loss) for _ in range(world_size)]
            gathered_accs = [torch.zeros_like(torch.tensor(local_acc, device=device)) for _ in range(world_size)]
            
            dist.all_gather(gathered_losses, local_loss)
            dist.all_gather(gathered_accs, torch.tensor(local_acc, device=device))
            
            global_loss = torch.mean(torch.stack(gathered_losses))
            global_acc = torch.mean(torch.stack(gathered_accs))
            
            batch_losses.append(global_loss.item())
            batch_accuracies.append(global_acc.item())
        
            #Log metrics (rank 0 only)
            if rank == 0: #and batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx}")
                print(f"Loss: {global_loss.item():.4f}, Acc: {global_acc.item():.4f}")
                print(f"Per-GPU losses: {[l.item() for l in gathered_losses]}")

            dist.barrier()  # Sync before next batch



        if rank == 0:
            avg_loss = sum(batch_losses) / len(batch_losses)
            avg_acc = sum(batch_accuracies) / len(batch_accuracies)
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Average Accuracy: {avg_acc:.4f}\n")
        val_loss, val_acc = validate(model, val_dl, modality_encoder, modality_proj, 
                                tokenizer_local, device, cfg, rank, world_size)
        
        if rank == 0:
            print(f"\nEpoch {epoch + 1} Validation:")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    if rank == 0:
        print(f"\nEpoch {epoch + 1} Validation:")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'proj_state_dict': modality_proj.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'train_loss': avg_loss,
            'train_acc': avg_acc,
            'config': cfg
        }
            
        checkpoint_dir = os.path.join('/p/scratch/hai_oneprot/bazarova1', 'checkpoints_pa', datetime.now().strftime("%Y-%m-%d_%H%M%S"))
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
        checkpoint_dir, 
            f'checkpoint_epoch{epoch+1}_valloss{val_loss:.4f}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
    
    
    # Clean up
    #dist.destroy_process_group()
    def get_validation_sample(val_dl, device):
        val_iter = iter(val_dl)
        modality_input, text_input, _ = next(val_iter)
        modality_input = modality_input.to(device)
        return modality_input, text_input

    if rank == 0:
        print("Generation phase:")

        val_modality_input, val_text_input = get_validation_sample(val_dl, device)
        val_modality_embs = encode_modality(val_modality_input, modality_encoder, modality_proj)

        print(f"- Final modality_embs shape: {val_modality_embs.shape}")
        
        generation_inputs = {
            "modality_embeds": [val_modality_embs],
            'prompt': cfg.dummy_prompt,
            'max_tgt_len': cfg.max_tgt_len,
            'top_p': cfg.top_p,
            'temperature': cfg.temperature,
            'modality_cache': cfg.modality_cache,
        }
        input_embeds = prepare_generation_embedding(generation_inputs, model, tokenizer_local,"modality_embeds")
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[2277], encounters=1)])
        outputs = model.module.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=generation_inputs['max_tgt_len'],
            top_p=generation_inputs['top_p'],
            temperature=generation_inputs['temperature'],
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )
        print(outputs.shape, outputs[0].shape, " outputs shape")
        output_text = tokenizer_local.decode(outputs[2][:-2], skip_special_tokens=True)
        print("Output:", output_text)
        print(f"Ground truth: {val_text_input[2]}")

    # structure_batch = next(iter(val_dl_structure))
    # input_embeds = model.get_input_embeddings()(structure_batch['input_ids'].to(device))

    # # Generate with same parameters
    # stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[2277], encounters=1)])
    # structure_outputs = model.module.generate(
    #     inputs_embeds=input_embeds,
    #     max_new_tokens=generation_inputs['max_tgt_len'],
    #     top_p=generation_inputs['top_p'],
    #     temperature=generation_inputs['temperature'],
    #     do_sample=True,
    #     use_cache=True,
    #     stopping_criteria=stopping_criteria,
    # )

    # # Decode and print results
    # structure_text = tokenizer_local.decode(structure_outputs[0][:-2], skip_special_tokens=True)
    # print("\nStructure Output:", structure_text)


        # Read CSV file
        csv_path = cfg.dataset.csv_file_path
        df = pd.read_csv(csv_path)

        # Get random row
        random_row = df.iloc[random.randint(0, len(df)-1)]

    # Print each column with label
    # print("Random sample analysis:")
    # print("-" * 50)
    # print(f"ID: {random_row['id']}")
    # print(f"MSA Path: {random_row['msa_path']}")
    # print(f"Function Text: {random_row['func_text'][:200]}...")  # First 200 chars
    # print(f"Structure Tokens: {random_row['struc_tokens'][:100]}...")  # First 100 chars
    # print(f"Structure: {random_row['structure']}")
    # print(f"Sequence: {random_row['sequence']}")
    # print(f"Pocket: {random_row['pocket']}")

    # modality_input, text_input = batch
    # modality_input = modality_input.to(device)
    # modality_embs = encode_modality(modality_input, modality_encoder, modality_proj).to(device)

        seq_tokenizer=AutoTokenizer.from_pretrained(cfg.dataset.seq_tokenizer)
        text_tokenizer=AutoTokenizer.from_pretrained(cfg.dataset.text_tokenizer)
        #struct_token_tokenizer=AutoTokenizer.from_pretrained(cfg.dataset.struct_token_tokenizer)
        new_tokens = ['p', 'y', 'n', 'w', 'r', 'q', 'h', 'g', 'd', 'l', 'v', 't', 'm', 'f', 's', 'a', 'e', 'i', 'k', 'c','#']
        struct_token_tokenizer = AutoTokenizer.from_pretrained(cfg.dataset.seq_tokenizer)
        struct_token_tokenizer.add_tokens(new_tokens)  

        seq_embedding = seq_tokenizer(random_row['sequence'], return_tensors="pt")["input_ids"].to(device)
        seq_embedding=encode_modality(seq_embedding, modality_encoder, modality_proj).to(device)

        text_embedding = text_tokenizer(random_row['func_text'], return_tensors="pt",max_length=512,truncation=True)["input_ids"].to(device)
        text_embedding=encode_modality(text_embedding, oneprot.network['text'].to(device), modality_proj).to(device)

        input_struct_token = "".join([s.replace("#", "") for s in random_row['struc_tokens']])
        input_tensor = seq_tokenizer(input_struct_token, return_tensors="pt")["input_ids"].to(device)
        struct_token_embedding =encode_modality(input_tensor, oneprot.network['struct_token'].to(device), modality_proj).to(device)



        with h5py.File('/p/scratch/hai_oneprot/merdivan1/pretrain_dataset/50ss/seqstruc.h5', 'r') as file:
            input_struct_graph=[protein_to_graph(random_row['structure'], '/p/scratch/hai_oneprot/merdivan1/pretrain_dataset/50ss/seqstruc.h5', 'non_pdb', 'A', pockets=False)]
            input_struct_graph = Batch.from_data_list(input_struct_graph).to(device)
            struct_graph_embedding=encode_modality(input_struct_graph, oneprot.network['struct_graph'].to(device), modality_proj).to(device)

        with h5py.File('/p/scratch/hai_oneprot/merdivan1/pretrain_dataset/50ss/pockets_100_residues.h5', 'r') as file:
            input_struct_graph=[protein_to_graph(random_row['pocket'], '/p/scratch/hai_oneprot/merdivan1/pretrain_dataset/50ss/seqstruc.h5', 'non_pdb', 'A', pockets=False)]
            input_struct_graph = Batch.from_data_list(input_struct_graph).to(device)
            pocket_embedding=encode_modality(input_struct_graph, oneprot.network['pocket'].to(device), modality_proj).to(device)

            generation_inputs = {
                "seq_embeds": [seq_embedding],
                "text_embeds": [text_embedding],
                "struct_token_embeds": [struct_token_embedding],
                "struct_graph_embeds": [struct_graph_embedding],
                "pocket_embeds": [pocket_embedding],
                'prompt': cfg.dummy_prompt,
                'max_tgt_len': cfg.max_tgt_len,
                'top_p': cfg.top_p,
                'temperature': cfg.temperature,
                'modality_cache': cfg.modality_cache,
            }

            for emb in generation_inputs.keys():
                if '_embeds' in emb:
                    print(emb," embs!!!!!!!!!!")
                    #print(generation_inputs[emb].shape)
                    input_embeds = prepare_generation_embedding(generation_inputs, model, tokenizer_local,emb)
                    print(input_embeds.shape," input embeds!!!!!!!!!!")
                    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[2277], encounters=1)])
                    outputs = model.module.generate(
                        inputs_embeds=input_embeds,
                        max_new_tokens=generation_inputs['max_tgt_len'],
                        top_p=generation_inputs['top_p'],
                        temperature=generation_inputs['temperature'],
                        do_sample=True,
                        use_cache=True,
                        stopping_criteria=stopping_criteria,
                    )
                    print(outputs.shape, outputs[0].shape, " outputs shape")
                    output_text = tokenizer_local.decode(outputs[0][:-2], skip_special_tokens=True)
                    print("Output:", output_text)
                    print(f"Ground truth: {random_row['func_text']}")

if __name__ == "__main__":
    main()