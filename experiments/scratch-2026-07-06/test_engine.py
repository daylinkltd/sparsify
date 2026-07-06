from sparsify.runtime.chat_generation import SparsifyEngine
from pathlib import Path

engine = SparsifyEngine(Path('/Volumes/projects/sparsify/models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit'), max_tokens=10)
print("Engine loaded. Generating...")
print(engine.generate("Hello, how are you?"))
