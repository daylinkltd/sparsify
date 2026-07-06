with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()
content = content.replace("self.max_cached_experts = 100", "self.max_cached_experts = 400")
with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.write(content)
