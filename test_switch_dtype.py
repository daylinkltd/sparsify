import mlx.core as mx
from mlx_lm.models.switch_layers import SwitchLinear

# Try passing float32 weight to gather_qmm
try:
    x = mx.ones((1, 2))
    idx = mx.array([0])
    w = mx.ones((1, 2, 2), dtype=mx.float32)
    scales = mx.ones((1, 2, 1), dtype=mx.float16)
    b = mx.ones((1, 2, 1), dtype=mx.float16)
    mx.gather_qmm(x, w, scales, b, idx, group_size=64)
except Exception as e:
    print("Expected error:", e)
