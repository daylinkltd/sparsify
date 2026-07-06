# Sparsify Documentation

## Guides

- [Profiler Guide](profiler-guide.md) — How to use the model profiler
- [Architecture](architecture.md) — System architecture overview

## Quick Reference

```bash
# Profile a GGUF model
sparsify profile-model path/to/model.gguf

# Show system capabilities
sparsify info

# View profile history
sparsify history

# JSON output
sparsify profile-model model.gguf --json

# Save to database
sparsify profile-model model.gguf --save

# Export to JSON file
sparsify profile-model model.gguf --export
```
