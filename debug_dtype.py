with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()

content = content.replace("self.model, self.tokenizer = load(str(model_path), lazy=True)",
    "self.model, self.tokenizer = load(str(model_path), lazy=True)\n        print('after load weight dtype:', self.model.model.layers[0].block_sparse_moe.switch_mlp.gate_proj.weight.dtype)\n")

with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.write(content)
