import re
with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()

content = content.replace("                print('PROXY CALL TRIGGERED!', flush=True)\n", "")

with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.write(content)

with open(".venv/lib/python3.12/site-packages/mlx_lm/models/mixtral.py", "r") as f:
    mx_content = f.read()

mx_content = mx_content.replace("    print('MixtralForCausalLM.__call__ self.model type:', type(self.model)); out = self.model(inputs, cache, input_embeddings)", "        out = self.model(inputs, cache, input_embeddings)")

with open(".venv/lib/python3.12/site-packages/mlx_lm/models/mixtral.py", "w") as f:
    f.write(mx_content)
