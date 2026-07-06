import mlx.core as mx
import mlx.nn as nn

class MyMod(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.zeros((2, 2))
    def __call__(self, x):
        return x @ self.weight

mod = MyMod()
x = mx.ones((2, 2))
# First token graph
mod.weight = mx.ones((2, 2)) * 5
y1 = mod(x)

# Before evaluating y1, we change mod.weight!
mod.weight = mx.ones((2, 2)) * 10
y2 = mod(x)

print("y1:\n", y1)
print("y2:\n", y2)
