import re
with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()

content = re.sub(r'self\.model\.model = PagingModelProxy\(self\.model\.model\)', 'self.model.model = PagingModelProxy(self.model.model)', content)
with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.write(content)
