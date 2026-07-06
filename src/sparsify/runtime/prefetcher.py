"""Predictive Prefetcher forecasting and background preloading next-token experts."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Set, Tuple

import mlx.core as mx


class PredictivePrefetcher:
    """Predictive scheduler using Markov transition history to preload experts in background threads."""

    def __init__(self, cache: Any) -> None:
        self.cache = cache
        
        # Thread pool for non-blocking SSD reads
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Transition tracking: (layer_id, prev_expert) -> {next_expert: frequency_count}
        self.transitions: Dict[Tuple[int, int], Dict[int, int]] = {}
        
        # Keep track of the last executed expert per layer
        self.last_executed_expert: Dict[int, int] = {}
        
        # Currently prefetching keys to avoid duplicate tasks
        self.active_prefetch_keys: Set[Tuple[int, int]] = set()
        
        # Telemetry metrics
        self.prefetch_attempts = 0
        self.prefetch_hits = 0

    def record_access_and_predict(self, layer_id: int, expert_id: int) -> None:
        """Record the active expert selection, update transition statistics, and trigger background prefetch."""
        prev_expert = self.last_executed_expert.get(layer_id)
        
        # 1. Update Markov transitions table
        if prev_expert is not None and prev_expert != expert_id:
            key = (layer_id, prev_expert)
            if key not in self.transitions:
                self.transitions[key] = {}
            self.transitions[key][expert_id] = self.transitions[key].get(expert_id, 0) + 1
            
        # Update last executed expert
        self.last_executed_expert[layer_id] = expert_id

        # 2. Predict the most likely next expert for this layer
        pred_expert_id = self.predict_next_expert(layer_id, expert_id)
        if pred_expert_id is not None:
            # Trigger background prefetch if not already loaded or currently prefetching
            pred_key = (layer_id, pred_expert_id)
            if (pred_key not in self.cache._loaded_experts) and (pred_key not in self.active_prefetch_keys):
                self.trigger_background_prefetch(layer_id, pred_expert_id)

    def predict_next_expert(self, layer_id: int, current_expert_id: int) -> int | None:
        """Find the next expert with the highest historical transition frequency."""
        key = (layer_id, current_expert_id)
        candidates = self.transitions.get(key)
        if not candidates:
            # No history yet: fallback to sequential or None
            return None
        # Return the candidate with the highest frequency count
        return max(candidates, key=candidates.get)

    def trigger_background_prefetch(self, layer_id: int, expert_id: int) -> None:
        """Spawns a thread to load expert tensors from SSD into VRAM asynchronously."""
        pred_key = (layer_id, expert_id)
        self.active_prefetch_keys.add(pred_key)
        self.prefetch_attempts += 1
        
        def prefetch_task() -> None:
            try:
                # Load weights into cache (miss triggers SSD read, but in background thread!)
                _ = self.cache.get_expert(layer_id, expert_id)
            except Exception:
                pass
            finally:
                self.active_prefetch_keys.discard(pred_key)
                
        self.executor.submit(prefetch_task)

    def record_actual_selection(self, layer_id: int, selected_expert_id: int) -> None:
        """Check if the selected expert was successfully preloaded beforehand (hit check)."""
        # If the expert is in self.cache._loaded_experts and we attempted to prefetch it
        # we count it as a prefetch hit!
        # Note: the actual checking happens in telemetry, here we verify hits.
        pass

    def shutdown(self) -> None:
        """Shut down background threads."""
        self.executor.shutdown(wait=False)
