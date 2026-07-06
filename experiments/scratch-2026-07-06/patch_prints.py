with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()

# Add a print before and after layer eval
old = """                    # 4. Evaluate layer (builds the graph using the small_tensor references)
                    h = layer(h, mask, c)"""
new = """                    # 4. Evaluate layer (builds the graph using the small_tensor references)
                    print(f"Evaluating layer {i}...", flush=True)
                    h = layer(h, mask, c)
                    print(f"Layer {i} graph built.", flush=True)"""
content = content.replace(old, new)

old_eval = """                    if c is not None and hasattr(c, "state"):
                        mx.eval(h)
                        mx.eval(c.state)
                        
                    _ = mx.sum(h).item() # Force CPU-GPU sync"""
new_eval = """                    if c is not None and hasattr(c, "state"):
                        mx.eval(h)
                        mx.eval(c.state)
                        
                    print(f"Forcing eval layer {i}...", flush=True)
                    _ = mx.sum(h).item() # Force CPU-GPU sync
                    print(f"Eval layer {i} done.", flush=True)"""
content = content.replace(old_eval, new_eval)

with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.write(content)
