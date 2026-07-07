import mlx.core as mx
import mlx.nn as nn

class MyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.zeros((2, 2))

mod = MyLinear()
print("Before setattr, mod.weight:", mod.weight, "mod['weight']:", mod['weight'])
new_weight = mx.ones((2, 2)) * 3
setattr(mod, 'weight', new_weight)
print("After setattr, mod.weight:", mod.weight, "mod['weight']:", mod['weight'])
