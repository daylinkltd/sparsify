with open(".venv/lib/python3.12/site-packages/mlx_lm/models/mixtral.py", "r") as f:
    content = f.read()
    
content = content.replace(
    "out = self.model(inputs, cache, input_embeddings)",
    "print('MixtralForCausalLM.__call__ self.model type:', type(self.model)); out = self.model(inputs, cache, input_embeddings)"
)

with open(".venv/lib/python3.12/site-packages/mlx_lm/models/mixtral.py", "w") as f:
    f.write(content)
