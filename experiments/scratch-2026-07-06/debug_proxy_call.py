with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()

content = content.replace("def __call__(self, inputs, cache=None, input_embeddings=None, **kwargs):", "def __call__(self, inputs, cache=None, input_embeddings=None, **kwargs):\n                print('PROXY CALL TRIGGERED!', flush=True)")

with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.write(content)
