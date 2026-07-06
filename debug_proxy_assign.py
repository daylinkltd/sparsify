with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()

content = content.replace(
    "        print('SET PROXY:', type(self.model), type(getattr(self.model, 'model')), type(self.model.model), flush=True)",
    "        print('IS PROXY?', isinstance(self.model.model, PagingModelProxy), type(self.model.model), self.model.model.__class__, flush=True)"
)

with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.write(content)
