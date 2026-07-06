with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()

content = content.replace("                for i, layer in enumerate(self.original_model.layers):", "                print(f'Starting forward pass for {inputs.shape}', flush=True)\n                for i, layer in enumerate(self.original_model.layers):")
content = content.replace("                    # For each layer, load the needed experts, evaluate, then unload", "                    print(f'Evaluating layer {i}', flush=True)\n                    # For each layer, load the needed experts, evaluate, then unload")

with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.write(content)
