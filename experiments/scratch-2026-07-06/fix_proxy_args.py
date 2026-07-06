with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()

# Fix proxy init to accept them
old_init = "            def __init__(self, model):"
new_init = "            def __init__(self, model, weight_map, safetensor_handles):"
content = content.replace(old_init, new_init)

old_proxy_inst = "        self.model.model = PagingModelProxy(self.model.model)"
new_proxy_inst = "        self.model.model = PagingModelProxy(self.model.model, self.weight_map, self.safetensor_handles)"
content = content.replace(old_proxy_inst, new_proxy_inst)

with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.write(content)
