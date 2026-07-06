import mlx.core as mx
import mlx.nn as nn
import gc

class Test(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(1000, 1000, bias=False)

t = Test()
print("Initial weights:", t.gate_proj.weight.shape)
big_array = mx.zeros((1000, 1000))
t.update({"gate_proj": {"weight": big_array}})
print("Updated weights:", t.gate_proj.weight.shape)
print("Is big_array?", t.gate_proj.weight is big_array)

dummy = mx.zeros((0,))
t.update({"gate_proj": {"weight": dummy}})
print("Dummy weights:", t.gate_proj.weight.shape)
print("Is dummy?", t.gate_proj.weight is dummy)
