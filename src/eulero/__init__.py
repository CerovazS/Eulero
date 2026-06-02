"""eulero - plug & play complex-valued neural network building blocks for PyTorch.

The building blocks live under :mod:`eulero.nn`. Example::

    import torch
    from eulero.nn import CMultiHeadedAttention, ComplexRMSNorm, get_activation

    act = get_activation("CReLU", is_complex=True)
"""

__version__ = "0.1.0"

from . import nn, utils  # noqa: F401

__all__ = ["nn", "utils", "__version__"]
