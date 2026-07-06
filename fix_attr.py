with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()

content = content.replace("for layer in self.model.model.original_model.layers:", "for layer in self.model.model.layers:")

with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.write(content)
print("Fixed attr!")
