"""LRU Cache for dynamic loading and eviction of Mixture-of-Experts (MoE) weights."""
from __future__ import annotations

import gc
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Tuple
import mlx.core as mx

from sparsify.utils.config import SPARSIFY_DIR


class ExpertLRUCache:
    """Bounded cache to manage loaded MoE expert weights in memory.

    Enforces that at most `max_active_experts` are in memory simultaneously,
    evicting the least recently used experts and cleaning up metal/host caches.
    """

    def __init__(self, max_active_experts: int = 8, experts_dir: Path | None = None) -> None:
        self.max_active_experts = max_active_experts
        self.experts_dir = experts_dir or (SPARSIFY_DIR / "experts")
        self.experts_dir.mkdir(parents=True, exist_ok=True)
        
        # OrderedDict mapping (layer_idx, expert_idx) -> weights dict
        self._cache: OrderedDict[Tuple[int, int], Dict[str, mx.array]] = OrderedDict()
        
        # Tracking metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get_expert(self, layer_idx: int, expert_idx: int) -> Dict[str, mx.array]:
        """Retrieve expert weights from memory cache, or load dynamically from disk."""
        key = (layer_idx, expert_idx)
        
        if key in self._cache:
            self.hits += 1
            # Move to end to mark as most recently used
            self._cache.move_to_end(key)
            return self._cache[key]
            
        self.misses += 1
        
        # Bounded memory: evict if full
        if len(self._cache) >= self.max_active_experts:
            self.evict_lru()
            
        # Load from disk
        path = self.experts_dir / f"layer_{layer_idx}_expert_{expert_idx}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Expert weights file not found: {path}")
            
        weights = mx.load(str(path))
        self._cache[key] = weights
        return weights

    def evict_lru(self) -> None:
        """Evict the least recently used expert weights and reclaim VRAM/RAM."""
        if not self._cache:
            return
            
        key, weights = self._cache.popitem(last=False)
        self.evictions += 1
        
        # Clear references to trigger garbage collection
        del weights
        
        # Reclaim Metal unified memory buffers immediately
        mx.clear_cache()
        gc.collect()

    def clear(self) -> None:
        """Clear all active experts from memory."""
        self._cache.clear()
        mx.clear_cache()
        gc.collect()

    @property
    def active_count(self) -> int:
        """Get the number of currently loaded experts in memory."""
        return len(self._cache)
