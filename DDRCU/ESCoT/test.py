from huggingface_hub import hf_hub_download

# config.json 다운로드
config_path = hf_hub_download(repo_id="meta-llama/Llama-2-7b-chat-hf", filename="model-00001-of-00002.safetensors")

# pytorch_model.bin 다운로드
model_path = hf_hub_download(repo_id="meta-llama/Llama-2-7b-chat-hf", filename="pytorch_model-00001-of-00002.bin")
	
	