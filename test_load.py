import time
import mlx.core as mx
from pathlib import Path

path = Path('models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit/model-00001-of-00013.safetensors')
t0 = time.time()
for _ in range(32):
    s = mx.load(str(path))
    s.clear()
print(f"32 mx.load() took {time.time()-t0:.2f}s")
