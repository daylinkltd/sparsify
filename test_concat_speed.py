import time
import mlx.core as mx

# Create dummy tensors on CPU/GPU
t0 = time.time()
tensors = [mx.random.uniform(shape=(1, 14336, 512)).astype(mx.uint32) for _ in range(8)]
mx.eval(tensors)
print(f"Init took: {time.time()-t0:.2f}s")

t0 = time.time()
concat = mx.concatenate([tensors[0], tensors[2]], axis=0)
mx.eval(concat)
print(f"Concat 1 took: {time.time()-t0:.2f}s")

t0 = time.time()
concat = mx.concatenate([tensors[1], tensors[3]], axis=0)
mx.eval(concat)
print(f"Concat 2 took: {time.time()-t0:.2f}s")
