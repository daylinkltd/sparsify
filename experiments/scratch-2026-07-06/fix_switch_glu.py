with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()

import_str = """
# Monkey patch SwitchGLU to support dynamic index mapping
from mlx_lm.models.switch_layers import SwitchGLU
if not hasattr(SwitchGLU, "_original_call"):
    SwitchGLU._original_call = SwitchGLU.__call__
    def _mapped_call(self, x, indices, **kwargs):
        if hasattr(self, "active_mapping"):
            indices = self.active_mapping[indices]
        return self._original_call(x, indices, **kwargs)
    SwitchGLU.__call__ = _mapped_call
"""

if "Monkey patch SwitchGLU" not in content:
    content = content.replace("class SparsifyEngine:", import_str + "\nclass SparsifyEngine:")

# In PagingModelProxy.__call__
old_patch = """                        if hasattr(moe_block, "switch_mlp") and hasattr(moe_block.switch_mlp, "__call__"):
                            original_switch_mlp_call = moe_block.switch_mlp.__call__
                            
                            def mapped_switch_mlp_call(x, inds_args, *args, **kw):
                                mapped_inds = mapping_mx[inds_args]
                                return original_switch_mlp_call(x, mapped_inds, *args, **kw)
                                
                            moe_block.switch_mlp.__call__ = mapped_switch_mlp_call"""
new_patch = """                        if hasattr(moe_block, "switch_mlp"):
                            moe_block.switch_mlp.active_mapping = mapping_mx"""
content = content.replace(old_patch, new_patch)

old_restore = """                        if original_switch_mlp_call is not None:
                            moe_block.switch_mlp.__call__ = original_switch_mlp_call"""
new_restore = """                        if hasattr(moe_block, "switch_mlp") and hasattr(moe_block.switch_mlp, "active_mapping"):
                            del moe_block.switch_mlp.active_mapping"""
content = content.replace(old_restore, new_restore)

with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.write(content)
