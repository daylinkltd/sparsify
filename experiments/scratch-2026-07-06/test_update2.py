import mlx.core as mx
import mlx.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = mx.zeros((10,), dtype=mx.uint32)

m = M()
print("original:", m.x.dtype)
m.update({'x': mx.zeros((10,), dtype=mx.float32)})
print("after float32 update:", m.x.dtype)
m.update({'x': mx.zeros((10,), dtype=mx.uint32)})
print("after uint32 update:", m.x.dtype)
