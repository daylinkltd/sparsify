import mlx.core as mx
import numpy as np

# create array
a = mx.ones((1000, 1000))
mx.eval(a)
# copy
b = mx.array(np.array(a))
print(b.shape)
