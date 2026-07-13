"""Byte-budgeted expert cache with deterministic LRU eviction for PyTorch."""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Tuple

import torch

Key = Tuple[int, int]  # (group_uid, expert_idx)
Entry = Dict[str, Dict[str, torch.Tensor]]  # proj_name -> param_name -> slice
RawEntry = Dict[str, Dict[str, tuple]]  # proj -> param -> (bytes, shape, dtype)

_IO_WORKERS = int(os.environ.get("SPARSIFY_IO_WORKERS", "8"))


class PyTorchExpertCache:
    """LRU cache of expert weight slices, bounded by a byte budget.

    The budget bounds bytes held *by the cache* in VRAM/RAM.
    """

    def __init__(self, store, budget_bytes: int) -> None:
        self.store = store
        self.budget_bytes = budget_bytes
        self._entries: "OrderedDict[Key, Entry]" = OrderedDict()
        self._sizes: Dict[Key, int] = {}
        self.used_bytes = 0
        self._pool = ThreadPoolExecutor(max_workers=_IO_WORKERS,
                                        thread_name_prefix="sparsify-torch-io")

        # Staging: raw prefetched bytes, bounded, consumed on first use.
        self._staged: "OrderedDict[Key, RawEntry]" = OrderedDict()
        self._staged_bytes = 0
        self._staging_lock = threading.Lock()
        self.staging_limit_bytes = 512 * 1024 * 1024

        # Speculative prefetch runs on its own small pool
        self._prefetch_pool = ThreadPoolExecutor(max_workers=3,
                                                 thread_name_prefix="sparsify-torch-prefetch")
        self._prefetch_inflight = 0

        # Telemetry
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.staged_hits = 0
        self.prefetched = 0

    # ── fetch paths ────────────────────────────────────────────────────

    def _fetch_raw(self, group, expert_idx: int) -> RawEntry:
        """Read one expert's raw bytes."""
        raw: RawEntry = {}
        for proj_name, sources in group.proj_sources.items():
            tensors = {}
            for param, (kind, ref) in sources.items():
                if kind == "stacked":
                    tensors[param] = self.store.read_expert_slice_raw(ref, expert_idx)
                else:  # "per_expert": one on-disk tensor per expert
                    b, shape, dtype = self.store.read_tensor_raw(ref[expert_idx])
                    tensors[param] = (b, [1] + shape, dtype)
            raw[proj_name] = tensors
        return raw

    def _wrap(self, raw: RawEntry) -> Entry:
        """Raw bytes -> PyTorch tensors."""
        return {
            proj: {param: self.store.wrap_raw(*t) for param, t in tensors.items()}
            for proj, tensors in raw.items()
        }

    def get_experts(self, group, expert_ids: Iterable[int]) -> Dict[int, Entry]:
        """Return entries for expert_ids of group, reading misses in parallel."""
        expert_ids = list(expert_ids)
        needed = {(group.uid, e) for e in expert_ids}
        out: Dict[int, Entry] = {}
        missing: List[int] = []

        for e in expert_ids:
            entry = self._entries.get((group.uid, e))
            if entry is not None:
                self.hits += 1
                self._entries.move_to_end((group.uid, e))
                out[e] = entry
            else:
                self.misses += 1
                missing.append(e)

        if missing:
            raws: Dict[int, RawEntry] = {}
            to_read: List[int] = []
            with self._staging_lock:
                for e in missing:
                    staged = self._staged.pop((group.uid, e), None)
                    if staged is not None:
                        self._staged_bytes -= _raw_nbytes(staged)
                        self.staged_hits += 1
                        raws[e] = staged
                    else:
                        to_read.append(e)
            if len(to_read) == 1:
                raws[to_read[0]] = self._fetch_raw(group, to_read[0])
            elif to_read:
                futures = {e: self._pool.submit(self._fetch_raw, group, e)
                           for e in to_read}
                for e, fut in futures.items():
                    raws[e] = fut.result()

            for e in missing:  # deterministic insertion order
                entry = self._wrap(raws[e])
                key = (group.uid, e)
                size = _entry_nbytes(entry)
                self._entries[key] = entry
                self._sizes[key] = size
                self.used_bytes += size
                out[e] = entry
            self._evict_to_budget(protect=needed)
        return out

    # ── prefetch staging (called from any thread) ──────────────────────

    def prefetch_async(self, group, expert_ids: Iterable[int]) -> None:
        """Stage expert_ids in the background, best-effort."""
        if self._prefetch_inflight >= 6:
            return
        ids = [e for e in expert_ids
               if (group.uid, e) not in self._entries
               and (group.uid, e) not in self._staged]
        if ids:
            self._prefetch_inflight += 1
            self._prefetch_pool.submit(self._prefetch_safe, group, ids)

    def _prefetch_safe(self, group, ids) -> None:
        try:
            self.prefetch_raw(group, ids)
        except Exception:
            pass  # best-effort
        finally:
            self._prefetch_inflight -= 1

    def prefetch_raw(self, group, expert_ids: Iterable[int]) -> None:
        """Read experts' raw bytes into staging if not already resident."""
        for e in expert_ids:
            key = (group.uid, e)
            if key in self._entries:
                continue
            with self._staging_lock:
                if key in self._staged:
                    continue
            raw = self._fetch_raw(group, e)
            size = _raw_nbytes(raw)
            with self._staging_lock:
                if key in self._staged or key in self._entries:
                    continue
                self._staged[key] = raw
                self._staged_bytes += size
                self.prefetched += 1
                while self._staged_bytes > self.staging_limit_bytes and self._staged:
                    _, dropped = self._staged.popitem(last=False)
                    self._staged_bytes -= _raw_nbytes(dropped)

    # ── eviction ───────────────────────────────────────────────────────

    def _evict_to_budget(self, protect: set) -> None:
        while self.used_bytes > self.budget_bytes:
            victim = None
            for key in self._entries:  # in LRU order
                if key not in protect:
                    victim = key
                    break
            if victim is None:
                break
            self._entries.pop(victim)
            self.used_bytes -= self._sizes.pop(victim)
            self.evictions += 1

    def close(self) -> None:
        """Shut down I/O pools."""
        self._prefetch_pool.shutdown(wait=False, cancel_futures=True)
        self._pool.shutdown(wait=False, cancel_futures=True)

    def stats(self) -> Dict[str, float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "evictions": self.evictions,
            "staged_hits": self.staged_hits,
            "prefetched": self.prefetched,
            "resident_experts": len(self._entries),
            "resident_bytes": self.used_bytes,
            "budget_bytes": self.budget_bytes,
        }


def _entry_nbytes(entry: Entry) -> int:
    return sum(t.numel() * t.element_size() for proj in entry.values() for t in proj.values())


def _raw_nbytes(raw: RawEntry) -> int:
    return sum(len(t[0]) for proj in raw.values() for t in proj.values())
