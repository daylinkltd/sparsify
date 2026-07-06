with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()

# Fix proxy init to accept them
old_init = "    def __init__(self, original_model, tokenizer, expert_cache, expert_cache_lru, max_cached_experts):"
new_init = "    def __init__(self, original_model, tokenizer, expert_cache, expert_cache_lru, max_cached_experts, weight_map, safetensor_handles):"
content = content.replace(old_init, new_init)

old_init_body = """        self.expert_cache = expert_cache
        self.expert_cache_lru = expert_cache_lru
        self.max_cached_experts = max_cached_experts"""

new_init_body = """        self.expert_cache = expert_cache
        self.expert_cache_lru = expert_cache_lru
        self.max_cached_experts = max_cached_experts
        self.weight_map = weight_map
        self.safetensor_handles = safetensor_handles"""
content = content.replace(old_init_body, new_init_body)

old_proxy_inst = """        self.proxy = PagingModelProxy(
            self.model.model, 
            self.tokenizer, 
            self.expert_cache, 
            self.expert_cache_lru, 
            self.max_cached_experts
        )"""

new_proxy_inst = """        self.proxy = PagingModelProxy(
            self.model.model, 
            self.tokenizer, 
            self.expert_cache, 
            self.expert_cache_lru, 
            self.max_cached_experts,
            self.weight_map,
            self.safetensor_handles
        )"""
content = content.replace(old_proxy_inst, new_proxy_inst)

with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.write(content)
