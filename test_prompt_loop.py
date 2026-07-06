import time
from pathlib import Path
import mlx.core as mx

t0 = time.time()
path = Path('models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit')

for i in range(100):
    for f in ["model-00001-of-00014.safetensors"]:
        shard = mx.load(str(path / f))
        
print(f"Time for 100 loops of mx.load: {time.time()-t0:.4f}s")
