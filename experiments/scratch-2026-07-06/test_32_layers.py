import mlx.core as mx
import mlx_lm
import time
import sys

mx.reset_peak_memory()
model, tokenizer = mlx_lm.load('/Volumes/projects/sparsify/models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit', lazy=True)
print('Initial active:', mx.get_active_memory() / 1e9, 'GB')
sys.stdout.flush()

compiled_layers = [mx.compile(layer) for layer in model.model.layers]

# dummy input
h = mx.random.normal((1, 1, 4096))
mask = None
cache = [None]*32

print("Running 32 layers...")
sys.stdout.flush()

for i, layer in enumerate(compiled_layers):
    t0 = time.time()
    h = layer(h, mask=mask, cache=cache[i])
    mx.eval(h)
    mx.metal.clear_cache()
    print(f"Layer {i} run in {time.time()-t0:.2f}s | Active: {mx.get_active_memory()/1e9:.2f}GB | Peak: {mx.get_peak_memory()/1e9:.2f}GB")
    sys.stdout.flush()

print('Final active:', mx.get_active_memory() / 1e9, 'GB')
