from sparsify.runtime.chat_generation import SparsifyEngine
import time

engine = SparsifyEngine("models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit")
prompt = "Hello"
print("Starting stream...")
for text, telemetry in engine.generate_stream(prompt):
    print(f"YIELDED text: {repr(text)}, tokens: {telemetry['n_tokens']}")
