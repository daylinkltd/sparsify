import mlx.core as mx
import mlx.nn as nn

class MyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.zeros((2, 2))
        self.freeze()

mod = MyLinear()
# Update weight using update()
mod.update({"weight": mx.zeros((0,))})
print("After update, mod.weight:", mod.weight, "mod['weight']:", mod['weight'])

# Now use setattr
setattr(mod, 'weight', mx.ones((2, 2)) * 3)
print("After setattr, mod.weight:", mod.weight, "mod['weight']:", mod['weight'])
