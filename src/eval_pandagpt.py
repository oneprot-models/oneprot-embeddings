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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from downstream.pandagpt.galactica_utils import StoppingCriteriaSub, define_prompt_wrap, encode_modality, process_batch_instance, prepare_generation_embedding

@hydra.main(version_base="1.3", config_path="../configs", config_name="panda_gpt.yaml")
def main(cfg):
    rank = int(os.environ.get('SLURM_PROCID', '0'))
    world_size = int(os.environ.get('SLURM_NTASKS', '1'))
    local_rank = int(os.environ.get('SLURM_LOCALID', '0'))
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '12345')

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # Initialize the process group
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

    # Load Galactica model
    model_path = '/p/scratch/hai_oneprot/huggingface/models/facebook/galactica-125m'
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=cfg.lora_r, 
        lora_alpha=cfg.lora_alpha, 
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules
    )

    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model = DDP(model, device_ids=[local_rank], bucket_cap_mb=25)

    # Load OneProt and freeze all parameters
    model_config_path = '/p/project1/hai_oneprot/bazarova1/oneprot-refined/logs/train/runs/2024-11-05_16-37-44/config.yaml'
    model_cfg = OmegaConf.load(model_config_path)
    oneprot = hydra.utils.instantiate(model_cfg.model)
    oneprot.load_state_dict(torch.load('/p/scratch/hai_oneprot/checkpoints_refined_111024/2024-11-05_19-20-45/epoch_043_28400.ckpt')["state_dict"])
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
    modality_proj = DDP(modality_proj, device_ids=[local_rank], bucket_cap_mb=25)

    # Load training data for finetuning
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    modality_dl = DataLoader(
        dataset=datamodule.datasets[f"{cfg.modality_data_name}_train"],
        batch_size=32,
        num_workers=datamodule.num_workers,
        pin_memory=datamodule.pin_memory,
        collate_fn=datamodule.datasets[f"{cfg.modality_data_name}_train"].text_collate_fn,
        drop_last=True,
        shuffle=False,
        sampler=DistributedSampler(datamodule.datasets[f"{cfg.modality_data_name}_train"], shuffle=True, rank=rank, num_replicas=world_size)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    for epoch in range(cfg.num_epochs):
        for batch in modality_dl:
            modality_input, text_input = batch
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
        
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]    # [B, S-1]
            labels = targets[:, 2:]
            gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)    # [B*S]
            valid_mask = (labels != -100).reshape(-1)
            valid_tokens = gen_acc & valid_mask    # [B*S]
            gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        
            if rank == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss.item()}, Accuracy: {gen_acc}")

    # Clean up

    if rank == 0:
        generation_inputs = {
            "modality_embeds": [modality_embs],
            'prompt': cfg.dummy_prompt,
            'max_tgt_len': cfg.max_tgt_len,
            'top_p': cfg.top_p,
            'temperature': cfg.temperature,
            'modality_cache': cfg.modality_cache,
        }
        input_embeds = prepare_generation_embedding(generation_inputs, model, tokenizer_local)
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[2277], encounters=1)])
        outputs = model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=generation_inputs['max_tgt_len'],
            top_p=generation_inputs['top_p'],
            temperature=generation_inputs['temperature'],
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )
        output_text = tokenizer_local.decode(outputs[0][:-2], skip_special_tokens=True)
        print("Output:", output_text)
        
    dist.destroy_process_group()

if __name__ == "__main__":
    main()