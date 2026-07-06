with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()

content = content.replace("setattr(mod, attr, small_tensor)", "print(f'{attr} small_tensor dtype is {small_tensor.dtype}', flush=True); setattr(mod, attr, small_tensor)")

with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.write(content)
