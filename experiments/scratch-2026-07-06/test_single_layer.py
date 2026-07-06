import mlx.core as mx
import mlx_lm
import time
import sys

mx.reset_peak_memory()
model, tokenizer = mlx_lm.load('/Volumes/projects/sparsify/models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit', lazy=True)
print('Initial active:', mx.get_active_memory() / 1e9, 'GB')
sys.stdout.flush()

layer = model.model.layers[0]
run_layer = mx.compile(layer)

# dummy input
h = mx.random.normal((1, 1, 4096))
mask = None
cache = None

print("Compiling layer 0...")
sys.stdout.flush()
t0 = time.time()
out = run_layer(h, mask=mask, cache=cache)
mx.eval(out)
print(f"Compilation finished in {time.time()-t0:.2f}s")
sys.stdout.flush()

print("Running layer 0...")
sys.stdout.flush()
t0 = time.time()
out = run_layer(h, mask=mask, cache=cache)
mx.eval(out)
print(f"Run finished in {time.time()-t0:.2f}s")
sys.stdout.flush()

print('Final active:', mx.get_active_memory() / 1e9, 'GB')
