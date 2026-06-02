"""Minimal example: a complex-valued residual block.

The kind of stack used inside complex source-separation U-Nets / mask estimators:
    complex conv  ->  complex normalization  ->  magnitude-gated activation
with a residual connection, operating end-to-end on complex spectrogram features.

Run with:  python examples/complex_block.py
"""
import torch
from torch import nn

from eulero.nn import ComplexRMSNorm
from eulero.nn.activations import ModReLU2d
from eulero.nn.conv import SConv2d


class ComplexResidualBlock(nn.Module):
    """(B, C, F, T) complex -> (B, C, F, T) complex."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = SConv2d(channels, channels, kernel_size, is_complex=True)
        self.conv2 = SConv2d(channels, channels, kernel_size, is_complex=True)
        # ComplexRMSNorm normalizes over the trailing (channel) dim, so we move
        # channels last for the norm and back afterwards.
        self.norm = ComplexRMSNorm(channels)
        self.act = ModReLU2d(channels=channels)

    def _norm_cl(self, x):  # x: (B, C, F, T) -> normalize over C
        x = x.permute(0, 2, 3, 1)          # (B, F, T, C)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)       # (B, C, F, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self._norm_cl(self.conv1(x)))
        h = self.conv2(h)
        return x + h


if __name__ == "__main__":
    torch.manual_seed(0)
    block = ComplexResidualBlock(channels=8)
    # a fake complex spectrogram: (batch, channels, freq, time)
    x = torch.randn(2, 8, 32, 50, dtype=torch.complex64)
    try:
        y = block(x)
    except (RuntimeError, NotImplementedError) as exc:
        print(f"[skip] native complex conv2d unsupported on this backend: {exc}")
    else:
        print(f"input : {tuple(x.shape)}  dtype={x.dtype}")
        print(f"output: {tuple(y.shape)}  dtype={y.dtype}")
        assert y.shape == x.shape and y.is_complex()
        print("OK - complex residual block forward pass succeeded.")
