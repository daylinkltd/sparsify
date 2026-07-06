with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()

content = content.replace("                self.model.model = PagingModelProxy(self.model.model)", "        self.model.model = PagingModelProxy(self.model.model)")
with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.write(content)
