"""Byte-budgeted expert cache with deterministic LRU eviction.

An entry is one expert of one MoE block: every projection tensor
(gate/up/down × weight/scales/biases/bias) for that expert, fetched in a
single group of disk reads. Eviction granularity is the whole expert.

Correctness note: eviction only drops the cache's reference. Arrays already
handed to an in-flight forward pass stay alive through normal refcounting,
so eviction can never corrupt a running computation.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, Tuple

import mlx.core as mx

Key = Tuple[int, int]  # (group_uid, expert_idx)
Entry = Dict[str, Dict[str, mx.array]]  # proj_name -> param_name -> (1, ...) slice


class ExpertCache:
    """LRU cache of expert weight slices, bounded by a byte budget.

    The budget bounds bytes held *by the cache*. A single forward call may
    transiently hold up to one MoE block's active experts on top of that;
    that overshoot is measured and reported, never hidden.
    """

    def __init__(self, store, budget_bytes: int) -> None:
        self.store = store
        self.budget_bytes = budget_bytes
        self._entries: "OrderedDict[Key, Entry]" = OrderedDict()
        self._sizes: Dict[Key, int] = {}
        self.used_bytes = 0

        # Telemetry — all values measured.
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get_experts(self, group, expert_ids: Iterable[int]) -> Dict[int, Entry]:
        """Return entries for *expert_ids* of *group*, reading misses from disk.

        Experts requested in this call are protected from eviction triggered
        by this same call, so a budget smaller than one block's working set
        degrades to overshoot (measured) rather than thrashing mid-call.
        """
        needed = {(group.uid, e) for e in expert_ids}
        out: Dict[int, Entry] = {}
        for e in expert_ids:
            key = (group.uid, e)
            entry = self._entries.get(key)
            if entry is not None:
                self.hits += 1
                self._entries.move_to_end(key)
            else:
                self.misses += 1
                entry = self._load(group, e)
                size = _entry_nbytes(entry)
                self._entries[key] = entry
                self._sizes[key] = size
                self.used_bytes += size
                self._evict_to_budget(protect=needed)
            out[e] = entry
        return out

    def _load(self, group, expert_idx: int) -> Entry:
        entry: Entry = {}
        for proj_name, sources in group.proj_sources.items():
            tensors = {}
            for param, (kind, ref) in sources.items():
                if kind == "stacked":
                    tensors[param] = self.store.read_expert_slice(ref, expert_idx)
                else:  # "per_expert": one on-disk tensor per expert
                    tensors[param] = mx.expand_dims(
                        self.store.read_tensor(ref[expert_idx]), 0
                    )
            entry[proj_name] = tensors
        return entry

    def _evict_to_budget(self, protect: set) -> None:
        while self.used_bytes > self.budget_bytes:
            victim = None
            for key in self._entries:  # in LRU order
                if key not in protect:
                    victim = key
                    break
            if victim is None:
                break  # everything resident is needed by the current call
            self._entries.pop(victim)
            self.used_bytes -= self._sizes.pop(victim)
            self.evictions += 1

    def stats(self) -> Dict[str, float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "evictions": self.evictions,
            "resident_experts": len(self._entries),
            "resident_bytes": self.used_bytes,
            "budget_bytes": self.budget_bytes,
        }


def _entry_nbytes(entry: Entry) -> int:
    return sum(t.nbytes for proj in entry.values() for t in proj.values())
