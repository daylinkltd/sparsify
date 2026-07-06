with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()

content = content.replace("model_path / ", "Path(model_path) / ")
content = content.replace("model_path = str(model_path)", "") # don't overwrite if it was a path? Wait, it's just a string argument.

# Just to be safe, I will change the init patch to wrap model_path in Path()
with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.write(content)
