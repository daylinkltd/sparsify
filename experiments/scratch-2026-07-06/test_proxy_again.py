import mlx.nn as nn
class Proxy: pass
class Outer(nn.Module): pass
o = Outer()
o.model = nn.Module()
print(type(o.model))
o.model = Proxy()
print(type(o.model))
