import mlx.core as mx

w = mx.ones((2, 2)) * 5
x = mx.ones((2, 2))
y = x @ w

# Change w reference
w = mx.ones((2, 2)) * 100
print("y:", y)
