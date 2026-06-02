# =============================================================================
# eulero.nn - plug & play complex-valued (and real-compatible) building blocks.
#
# Building blocks for EuleroDec, a complex-valued RVQ-VAE for audio coding
# (https://arxiv.org/pdf/2601.17517), reusable in any PyTorch project.
#
# Importing ``eulero.nn`` eagerly exposes the most common modules. Specialised
# variants are always reachable from their submodules, e.g.
#     from eulero.nn.normalization.complex import ComplexBatchNorm1d
# =============================================================================

# Subpackages (reachable as eulero.nn.activations, eulero.nn.conv, ...)
from . import activations, attention, conv, embeddings, normalization, subsampling

# --- Activations (also populates eulero.nn.<Name>, used by get_activation) ----
from .activations import *  # noqa: F401,F403

# --- Normalization (real + complex, incl. reintegrated ComplexRMSNorm) --------
from .normalization import *  # noqa: F401,F403

# --- Positional embeddings (real + complex) -----------------------------------
from .embeddings import *  # noqa: F401,F403

# --- Convolutions (SConv1d/2d, NormConv*, NormLinear, transposed, ...) --------
from .conv import *  # noqa: F401,F403

# --- Core layers --------------------------------------------------------------
from .layers import (
    CLinear,
    ComplexDropout,
    PositionwiseFeedForward,
    MultiSequential,
    repeat,
)

# --- Complex patch merging / unmerging ----------------------------------------
from .complex_patch_merging import (
    PatchMergingLinearComplex,
    ConvPatchDownsampleComplex,
    PatchUnmergingLinearComplex,
    ConvPatchUpsampleComplex,
)

# --- Attention ----------------------------------------------------------------
from .attention import (
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
    MultiHeadSelfAttention,
    CMultiHeadedAttention,
)

# --- Recurrent ----------------------------------------------------------------
from .rnn import SLSTM
