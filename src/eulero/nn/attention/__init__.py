# =============================================================================
# Entrypoint for real and complex attention modules.
# =============================================================================
from .standard import (
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
    MultiHeadSelfAttention,
)
from .complex import CMultiHeadedAttention

__all__ = [
    "MultiHeadedAttention",
    "RelPositionMultiHeadedAttention",
    "MultiHeadSelfAttention",
    "CMultiHeadedAttention",
]
