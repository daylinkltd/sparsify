"""Memory Manager and Expert Cache implementing LRU, LFU, and ARC eviction policies."""
from __future__ import annotations

import gc
import time
from collections import OrderedDict
from typing import Any, Dict, List, Set, Tuple

import mlx.core as mx
from sparsify.runtime.registry import ExpertMetadata
from sparsify.runtime.storage import load_expert_tensors


class EvictionPolicy:
    """Base class for expert cache eviction policies."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity

    def access(self, key: Tuple[int, int]) -> Tuple[int, int] | None:
        """Record an access and return the key of any evicted expert, or None."""
        raise NotImplementedError()

    def record_hit(self, key: Tuple[int, int]) -> None:
        """Record a hit on an already loaded expert."""
        pass

    def remove(self, key: Tuple[int, int]) -> None:
        """Remove a key manually from the policy tracking."""
        pass


class LRUPolicy(EvictionPolicy):
    """Least Recently Used (LRU) Cache policy."""

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self.keys: OrderedDict[Tuple[int, int], None] = OrderedDict()

    def access(self, key: Tuple[int, int]) -> Tuple[int, int] | None:
        if key in self.keys:
            self.keys.move_to_end(key)
            return None

        evicted = None
        if len(self.keys) >= self.capacity:
            # Pop the first element (least recently used)
            evicted, _ = self.keys.popitem(last=False)

        self.keys[key] = None
        return evicted

    def record_hit(self, key: Tuple[int, int]) -> None:
        if key in self.keys:
            self.keys.move_to_end(key)

    def remove(self, key: Tuple[int, int]) -> None:
        self.keys.pop(key, None)


class LFUPolicy(EvictionPolicy):
    """Least Frequently Used (LFU) Cache policy."""

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self.counts: Dict[Tuple[int, int], int] = {}
        # Store insertion order for tie-breaking
        self.keys: OrderedDict[Tuple[int, int], None] = OrderedDict()

    def access(self, key: Tuple[int, int]) -> Tuple[int, int] | None:
        if key in self.keys:
            self.counts[key] = self.counts.get(key, 0) + 1
            self.keys.move_to_end(key)
            return None

        evicted = None
        if len(self.keys) >= self.capacity:
            # Find the key with the minimum access count
            min_count = min(self.counts.values())
            # Evict the least frequently used. Tie-break using insertion order (LRU)
            min_keys = [k for k in self.keys if self.counts.get(k, 0) == min_count]
            evicted = min_keys[0]
            self.keys.pop(evicted)
            self.counts.pop(evicted, None)

        self.keys[key] = None
        self.counts[key] = 1
        return evicted

    def record_hit(self, key: Tuple[int, int]) -> None:
        if key in self.keys:
            self.counts[key] = self.counts.get(key, 0) + 1
            self.keys.move_to_end(key)

    def remove(self, key: Tuple[int, int]) -> None:
        self.keys.pop(key, None)
        self.counts.pop(key, None)


class ARCPolicy(EvictionPolicy):
    """Adaptive Replacement Cache (ARC) self-tuning cache policy."""

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self.p: float = 0.0  # Partition target size of T1 vs T2
        
        # T1: active recently used in memory
        self.t1: OrderedDict[Tuple[int, int], None] = OrderedDict()
        # B1: ghost cache for recently used (evicted)
        self.b1: OrderedDict[Tuple[int, int], None] = OrderedDict()
        
        # T2: active frequently used in memory
        self.t2: OrderedDict[Tuple[int, int], None] = OrderedDict()
        # B2: ghost cache for frequently used (evicted)
        self.b2: OrderedDict[Tuple[int, int], None] = OrderedDict()

    def _replace(self, key: Tuple[int, int], in_b2: bool) -> Tuple[int, int] | None:
        # Check T1 capacity conditions
        if self.t1 and (len(self.t1) > self.p or (in_b2 and len(self.t1) == self.p)):
            # Evict from T1 to B1
            ev_key, _ = self.t1.popitem(last=False)
            self.b1[ev_key] = None
            return ev_key
        elif self.t2:
            # Evict from T2 to B2
            ev_key, _ = self.t2.popitem(last=False)
            self.b2[ev_key] = None
            return ev_key
        return None

    def access(self, key: Tuple[int, int]) -> Tuple[int, int] | None:
        # Cache hit in memory is handled by record_hit.
        # This access handles a cache miss in active memory (possibly in ghost caches).
        evicted = None
        c = self.capacity

        # Hit in B1 (ghost cache for recency)
        if key in self.b1:
            # Shift partition to favor recency
            self.p = min(c, self.p + max(1, len(self.b2) // max(1, len(self.b1))))
            evicted = self._replace(key, in_b2=False)
            self.b1.pop(key)
            self.t2[key] = None
            return evicted

        # Hit in B2 (ghost cache for frequency)
        elif key in self.b2:
            # Shift partition to favor frequency
            self.p = max(0, self.p - max(1, len(self.b1) // max(1, len(self.b2))))
            evicted = self._replace(key, in_b2=True)
            self.b2.pop(key)
            self.t2[key] = None
            return evicted

        # Absolute cache miss (not in memory, not in ghost caches)
        else:
            l1_size = len(self.t1) + len(self.b1)
            if l1_size == c:
                if len(self.t1) < c:
                    self.b1.popitem(last=False)
                    evicted = self._replace(key, in_b2=False)
                else:
                    ev_key, _ = self.t1.popitem(last=False)
                    evicted = ev_key
            elif l1_size < c:
                total_size = len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2)
                if total_size >= c:
                    if total_size == 2 * c:
                        self.b2.popitem(last=False)
                    evicted = self._replace(key, in_b2=False)
            
            self.t1[key] = None
            return evicted

    def record_hit(self, key: Tuple[int, int]) -> None:
        # Key hit in active memory
        if key in self.t1:
            self.t1.pop(key)
            self.t2[key] = None
            self.t2.move_to_end(key)
        elif key in self.t2:
            self.t2.move_to_end(key)

    def remove(self, key: Tuple[int, int]) -> None:
        self.t1.pop(key, None)
        self.t2.pop(key, None)
        self.b1.pop(key, None)
        self.b2.pop(key, None)


class MoeCache:
    """Active Memory Manager orchestrating expert loading and evictions."""

    def __init__(
        self,
        registry: Any,
        budget_bytes: int = 4 * 1024 * 1024 * 1024,  # Default 4GB
        policy_name: str = "lru",
    ) -> None:
        self.registry = registry
        self.budget_bytes = budget_bytes
        self.policy_name = policy_name.lower()
        
        # Registry mapping layer_id/expert_id to active memory arrays
        self._loaded_experts: Dict[Tuple[int, int], Dict[str, mx.array]] = {}
        
        # Bounded capacity based on budget size
        # We assume typical expert parameter size is ~3.14MB (e.g. 786,432 float32 parameters)
        # We can dynamically estimate expert size from the registry.
        self.expert_size_bytes = 3 * 1024 * 1024  # Default fallback 3MB
        self.capacity = max(2, budget_bytes // self.expert_size_bytes)
        
        # Instantiate eviction policy
        if self.policy_name == "lru":
            self.policy = LRUPolicy(self.capacity)
        elif self.policy_name == "lfu":
            self.policy = LFUPolicy(self.capacity)
        elif self.policy_name == "adaptive":
            self.policy = ARCPolicy(self.capacity)
        else:
            raise ValueError(f"Unknown cache policy: {self.policy_name}")

        # Metrics Telemetry
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.bytes_read_total = 0

    def adjust_capacity(self, expert_size_bytes: int) -> None:
        """Dynamically recalibrate cache capacity based on the actual expert file size."""
        self.expert_size_bytes = expert_size_bytes
        self.capacity = max(2, self.budget_bytes // expert_size_bytes)
        self.policy.capacity = self.capacity

    def get_expert(self, layer_id: int, expert_id: int) -> Dict[str, mx.array]:
        """Request expert layers, loading them from SSD on cache miss."""
        key = (layer_id, expert_id)

        if key in self._loaded_experts:
            self.hits += 1
            self.policy.record_hit(key)
            return self._loaded_experts[key]

        self.misses += 1
        
        # 1. Look up expert location metadata
        metadata = self.registry.get_expert(layer_id, expert_id)
        if metadata is None:
            raise ValueError(f"Expert L{layer_id} E{expert_id} is not registered.")
            
        # Dynamically adjust capacity based on actual expert size
        if metadata.size_bytes != self.expert_size_bytes:
            self.adjust_capacity(metadata.size_bytes)

        # 2. Trigger eviction if capacity reached
        evicted_key = self.policy.access(key)
        if evicted_key is not None:
            self.evict_expert(evicted_key[0], evicted_key[1])

        # 3. Load expert from SSD
        weights = load_expert_tensors(metadata)
        self._loaded_experts[key] = weights
        self.bytes_read_total += metadata.size_bytes

        return weights

    def evict_expert(self, layer_id: int, expert_id: int) -> None:
        """Evict the expert weights from memory, calling cleanup hooks."""
        key = (layer_id, expert_id)
        if key in self._loaded_experts:
            weights = self._loaded_experts.pop(key)
            self.policy.remove(key)
            self.evictions += 1
            
            # Clear references
            del weights
            
            # Reclaim Metal unified memory buffers immediately
            mx.clear_cache()
            gc.collect()

    def clear(self) -> None:
        """Clear all active experts from memory cache."""
        self._loaded_experts.clear()
        self.policy.remove(None)
        mx.clear_cache()
        gc.collect()

    @property
    def loaded_count(self) -> int:
        return len(self._loaded_experts)

    @property
    def active_memory_footprint_bytes(self) -> int:
        """Compute the actual byte footprint of all loaded expert parameters."""
        return sum(
            meta.size_bytes
            for (l, e), meta in self.registry.experts.items()
            if (l, e) in self._loaded_experts
        )
