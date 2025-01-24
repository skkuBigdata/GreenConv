import os
import torch
from transformers import AutoModelForCausalLM

def merge_llm_with_lora(base_model_path, lora_model_path, output_dir):
    """
    Merge a base model with LoRA weights and save the merged model.

    Args:
        base_model_path (str): Path to the pre-trained base model directory.
        lora_model_path (str): Path to the directory containing LoRA weights.
        output_dir (str): Path to save the merged model.
    """

    print("Loading the base model...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

    # Load the LoRA weights
    print("Loading the LoRA weights...")
    lora_weights = torch.load(os.path.join(lora_model_path, "pytorch_model.bin"))

    # Merge LoRA weights into the base model
    print("Merging LoRA weights into the base model...")
    for name, param in base_model.named_parameters():
        if name in lora_weights:
            print(f"Merging parameter: {name}")
            param.data += lora_weights[name].to(param.device)

    # Save
    print("Saving the merged model...")
    os.makedirs(output_dir, exist_ok=True)
    base_model.save_pretrained(output_dir)
    print(f"Merged model saved to {output_dir}")
