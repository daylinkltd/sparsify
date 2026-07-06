with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()

import re
# Remove the prints and eval lines
content = re.sub(r'print\(f"Evaluating layer.*?\n', '', content)
content = re.sub(r'print\(f"Layer.*?\n', '', content)
content = re.sub(r'print\(f"Forcing eval layer.*?\n', '', content)
content = re.sub(r'print\(f"Eval layer.*?\n', '', content)
content = re.sub(r' *mx\.eval\(h\)\n', '', content)
content = re.sub(r' *mx\.eval\(c\.state\)\n', '', content)
content = re.sub(r' *_ = mx\.sum\(h\)\.item\(\) # Force CPU-GPU sync\n', '', content)

with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.write(content)
