"""Fairy2 inference optimizations.

This module provides optimized inference for Fairy2i models:
- TableLookupInference: Multiplication-free inference using table lookup
"""

from fairy2.inference.table_lookup import TableLookupInference

__all__ = [
    "TableLookupInference",
]
