import os
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)

def download_model(model_name, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    # Downloading the model and tokenizer
    cache_dir = os.path.join(save_directory, model_name)
    if model_name in ["facebook/galactica-120b"]:
        AutoModelForCausalLM.from_pretrained(model_name).save_pretrained(cache_dir)
        AutoTokenizer.from_pretrained(model_name).save_pretrained(cache_dir)

    # elif model_name == 'meta-llama/Llama-2-7b-hf':
    #     device = "cuda"
    #     dtype = torch.bfloat16
    #     #AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, torch_dtype=dtype, device_map=device, token='HF_TOKEN_REMOVED').save_pretrained(cache_dir)
    # 
    #     trash_dir = '/p/scratch/hai_baylora/trash'
    #     AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, cache_dir=trash_dir).save_pretrained(cache_dir)
    #     AutoTokenizer.from_pretrained(model_name).save_pretrained(cache_dir)

    else:
        # Add other model download steps if needed
        # Add other model download steps if needed″
        pass

    print(f"Model {model_name} downloaded and saved to {save_directory}")

# Example usage
download_model('facebook/galactica-120b', '/p/scratch/hai_oneprot/huggingface/models')