with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()

content = content.replace("e_slice_mx = mx.array(e_slice)", "e_slice_mx = mx.array(e_slice, dtype=full_tensor.dtype)")

with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.write(content)
