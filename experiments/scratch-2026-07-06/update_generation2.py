with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()

# Add a print statement before safetensors open
old_str = "with safe_open(file_path, framework=\"numpy\") as f:"
new_str = "print(f'Loading {weight_name} expert {e}', flush=True)\n                                                        with safe_open(file_path, framework=\"numpy\") as f:"
content = content.replace(old_str, new_str)

with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.write(content)
