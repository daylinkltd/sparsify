import mlx.core as mx
import mlx_lm
from pathlib import Path
import time

mx.metal.set_cache_limit(0)  # old way
try:
    mx.metal.set_wired_limit(4 * 1024 * 1024 * 1024)
except Exception:
    pass

model_path = "models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit"
print("Loading model...")
model, tokenizer = mlx_lm.load(model_path)
print("Model loaded. RAM should be near 0.")

prompt = "Hello, what is 2+2?"
messages = [{"role": "user", "content": prompt}]
prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
prompt_tokens = tokenizer.encode(prompt_text)
input_tokens = mx.array(prompt_tokens)

print("Evaluating...")
t0 = time.time()
for response in mlx_lm.generate_step(input_tokens, model):
    mx.eval(response.token)
    print(tokenizer.decode([response.token.item()]), end="", flush=True)
print(f"\nDone in {time.time()-t0:.2f}s")
