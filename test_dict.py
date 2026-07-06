class A(dict):
    def __getattr__(self, name):
        if name in self: return self[name]
        return super().__getattribute__(name)
        
a = A()
a['model'] = 5
print(a.model)
a.model = 10
print(a.model)
