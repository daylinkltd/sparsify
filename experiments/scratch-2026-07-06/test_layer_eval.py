import mlx.core as mx
import mlx_lm
import time
import sys

mx.reset_peak_memory()
model, tokenizer = mlx_lm.load('/Volumes/projects/sparsify/models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit', lazy=True)
print('Initial active:', mx.get_active_memory() / 1e9, 'GB')
sys.stdout.flush()

prompt = 'Hi'
messages = [{'role': 'user', 'content': prompt}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
x = mx.array(tokenizer.encode(formatted))[None]

run_embed = mx.compile(model.model.embed_tokens)
compiled_layers = [mx.compile(layer) for layer in model.model.layers]
run_norm = mx.compile(model.model.norm)
run_head = mx.compile(model.lm_head)

def step(x, cache):
    if x.shape[1] > 1:
        mask = mlx_lm.models.base.create_attention_mask(x, cache[0])
    else:
        mask = None

    h = run_embed(x)
    mx.eval(h)
    
    for i, layer in enumerate(compiled_layers):
        h = layer(h, mask=mask, cache=cache[i])
        mx.eval(h)
        
    logits = run_head(run_norm(h))
    mx.eval(logits)
    return mx.argmax(logits[:, -1, :], axis=-1)[None]

cache = [None] * len(model.model.layers)
for i in range(5):
    t0 = time.time()
    x = step(x, cache)
    print(f'Step {i} ({time.time()-t0:.2f}s): {tokenizer.decode([x.item()])} | Active: {mx.get_active_memory() / 1e9:.2f} GB | Peak: {mx.get_peak_memory() / 1e9:.2f} GB')
    sys.stdout.flush()
print("Done!")
