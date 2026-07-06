import huggingface_hub.utils
from tqdm.rich import tqdm
huggingface_hub.utils.tqdm = tqdm
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit')
