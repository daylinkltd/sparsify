with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()

content = content.replace("setattr(mod, attr, small_tensor)", "print(f'{attr} small_tensor dtype: {small_tensor.dtype} vs {full_tensor.dtype}'); setattr(mod, attr, small_tensor)")

with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.write(content)
