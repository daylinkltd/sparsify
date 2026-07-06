with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()

# Change the loop inside __init__ to look at self.original_model.model.layers
content = content.replace("for i, layer in enumerate(self.original_model.layers):", "for i, layer in enumerate(self.original_model.model.layers):")

# Change the wrapper application
old_wrapper = "self.model.model = PagingModelProxy(self.model.model)"
new_wrapper = "self.model = PagingModelProxy(self.model)"
content = content.replace(old_wrapper, new_wrapper)

# Change the layers loop inside __call__
content = content.replace("len(self.original_model.layers)", "len(self.original_model.model.layers)")
content = content.replace("for i, layer in enumerate(self.original_model.layers):", "for i, layer in enumerate(self.original_model.model.layers):")

# In PagingModelProxy.__call__, we need to return what MixtralForCausalLM.__call__ does?
# NO! Wait! If PagingModelProxy runs the layers manually, it bypasses MixtralForCausalLM.__call__!
# MixtralForCausalLM.__call__ does:
# out = self.model(inputs, cache, input_embeddings)
# return self.lm_head(out)
# So PagingModelProxy SHOULD do that!
