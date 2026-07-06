import mlx.core as mx
from mlx_lm import load
model, tokenizer = load("models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit", lazy=True)

class PagingModelProxy:
    def __init__(self, m):
        self.original_model = m
    def __getattr__(self, name):
        return getattr(self.original_model, name)

proxy = PagingModelProxy(model.model)
model.model = proxy
print("Is proxy:", isinstance(model.model, PagingModelProxy))
print("type:", type(model.model))
