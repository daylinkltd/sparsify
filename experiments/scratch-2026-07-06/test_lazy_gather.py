import mlx.core as mx
import time

mx.reset_peak_memory()

# Create a huge dummy file
data = mx.random.normal((8, 4096, 14336), dtype=mx.float16)
mx.save_safetensors("dummy_experts.safetensors", {"weights": data})

print('Initial active:', mx.get_active_memory() / 1e9, 'GB')

# Lazy load
loaded = mx.load("dummy_experts.safetensors", return_metadata=False)
weights = loaded["weights"]

print('After lazy load active:', mx.get_active_memory() / 1e9, 'GB')

# Gather 2 experts
indices = mx.array([1, 4])
out = weights[indices]
mx.eval(out)

print('After gather active:', mx.get_active_memory() / 1e9, 'GB')
