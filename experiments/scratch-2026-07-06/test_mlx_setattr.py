import mlx.nn as nn

class MixtralModel(nn.Module):
    def __call__(self, x):
        print("MixtralModel __call__")
        return x

class MixtralForCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MixtralModel()
        
    def __call__(self, x):
        return self.model(x)

class PagingModelProxy:
    def __init__(self, m):
        self.original = m
    def __call__(self, x):
        print("Proxy __call__")
        return self.original(x)

outer = MixtralForCausalLM()
outer.model = PagingModelProxy(outer.model)
outer("test")
