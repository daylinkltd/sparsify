with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()

content = content.replace("for response in self._mlx_lm.stream_generate(", "print('TYPE:', type(self.model)); for response in self._mlx_lm.stream_generate(")

with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.write(content)
