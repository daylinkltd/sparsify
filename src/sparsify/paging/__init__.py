"""Sparsify expert paging — storage-backed execution for MoE models.

Public API:
    attach_paging(model, model_path, budget_bytes) -> PagingRuntime | None
"""
from sparsify.paging.surgery import attach_paging
from sparsify.paging.cache import ExpertCache
from sparsify.paging.store import SafetensorsExpertStore

__all__ = ["attach_paging", "ExpertCache", "SafetensorsExpertStore"]
