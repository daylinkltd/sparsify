with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "setattr(mod, attr, small_tensor)" in line:
        indent = line[:len(line) - len(line.lstrip())]
        lines.insert(i, indent + 'print(f"Attr: {attr}, dtype: {small_tensor.dtype}", flush=True)\n')
        break

with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.writelines(lines)
