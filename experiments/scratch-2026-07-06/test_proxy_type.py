from sparsify.runtime.chat_generation import SparsifyEngine
from pathlib import Path

engine = SparsifyEngine(Path("models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit"))
print("After init, self.model.model type:", type(engine.model.model))
