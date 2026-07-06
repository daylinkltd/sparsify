import mlx.core as mx
import mlx.nn as nn

class MyMod(nn.Module):
    def __call__(self, x):
        return x + 1

mod = MyMod()
print(mod(mx.array(5)))

orig_call = mod.__call__
def new_call(x):
    return orig_call(x * 2)

mod.__call__ = new_call
print(mod(mx.array(5)))
