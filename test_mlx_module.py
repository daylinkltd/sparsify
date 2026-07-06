import mlx.core as mx
import mlx.nn as nn

class MyMod(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.zeros((2, 2))

mod = MyMod()
print(mod["weight"])
mod.weight = mx.ones((2, 2))
print(mod["weight"])
