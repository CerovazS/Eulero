"""Shape/dtype smoke tests for the eulero complex-valued building blocks.

These run on CPU with torch.complex64. Blocks that depend on backend features
not always available on CPU (native complex convolution, flex_attention) are
guarded and skipped rather than failing the suite.
"""
import pytest
import torch

import eulero
from eulero import nn as enn


def _cplx(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def test_package_imports_and_version():
    assert isinstance(eulero.__version__, str)
    # the factory namespace must expose activation classes by name
    assert hasattr(enn, "CReLU")
    assert hasattr(enn, "ComplexRMSNorm")


def test_get_activation_factory():
    real_act = enn.get_activation("ReLU", is_complex=False)
    assert isinstance(real_act, torch.nn.ReLU)

    cplx_act = enn.get_activation("CReLU", is_complex=True)
    x = _cplx(2, 16)
    y = cplx_act(x)
    assert y.is_complex() and y.shape == x.shape

    # unknown name degrades to Identity rather than raising
    assert isinstance(enn.get_activation("DoesNotExist", is_complex=True), torch.nn.Identity)


@pytest.mark.parametrize("name", ["CReLU", "zReLU", "magReLU", "CSiLU", "CTanh", "CGLU", "CMish"])
def test_elementwise_complex_activations(name):
    act = getattr(enn, name)()
    x = _cplx(4, 8, 8)
    y = act(x)
    assert y.is_complex()


def test_channelwise_activations():
    x = _cplx(2, 32, 50)          # (B, C, L)
    y = enn.activations.ModReLU1d(channels=32)(x)
    assert y.shape == x.shape and y.is_complex()

    g = enn.ComplexGELU2d(channels=8)
    x2 = _cplx(2, 8, 5, 5)        # (B, C, H, W)
    assert g(x2).shape == x2.shape


def test_complex_rms_norm():
    norm = enn.ComplexRMSNorm(64)
    x = _cplx(3, 10, 64)
    y = norm(x)
    assert y.shape == x.shape and y.is_complex()
    # real input must raise
    with pytest.raises(TypeError):
        norm(torch.randn(3, 10, 64))


def test_complex_normalizations():
    from eulero.nn.normalization.complex import ComplexGroupNorm, ComplexConvLayerNorm2d

    x = _cplx(2, 16, 8, 8)                       # (B, C, H, W)
    gn = ComplexGroupNorm(num_channels=16, num_groups=4)
    assert gn(x).shape == x.shape

    ln = ComplexConvLayerNorm2d(num_channels=16)
    assert ln(x).shape == x.shape


def test_complex_positional_encoding():
    pe = enn.embeddings.ComplexPositionalEncoding(d_model=32, dropout_rate=0.0)
    x = _cplx(2, 20, 32)
    y = pe(x)
    assert y.is_complex() and y.shape[-1] == 32


def test_clinear_and_dropout():
    lin = enn.CLinear(16, 24)
    x = _cplx(4, 16)
    y = lin(x)
    assert y.shape == (4, 24) and y.is_complex()

    drop = enn.ComplexDropout(p=0.5)
    drop.train()
    assert drop(x).is_complex()


def test_patch_merging_halves_grid():
    merge = enn.PatchMergingLinearComplex(dim=16, out_dim=32)
    H = W = 8
    x = _cplx(2, H * W, 16)
    y, (H2, W2) = merge(x, (H, W))
    assert (H2, W2) == (4, 4)
    assert y.shape == (2, 16, 32) and y.is_complex()


def test_sconv2d_complex_optional():
    """Native complex conv may be unsupported on this backend; skip if so."""
    conv = enn.conv.SConv2d(4, 8, kernel_size=3, is_complex=True)
    x = _cplx(1, 4, 16, 16)
    try:
        y = conv(x)
    except (RuntimeError, NotImplementedError) as exc:
        pytest.skip(f"complex conv2d unsupported on this backend: {exc}")
    assert y.is_complex() and y.shape[1] == 8


def test_cmultiheadedattention_optional():
    attn = enn.CMultiHeadedAttention(n_head=4, n_feat=32, dropout_rate=0.0)
    x = _cplx(2, 12, 32)
    try:
        y = attn(x, x, x)
    except (RuntimeError, NotImplementedError) as exc:
        pytest.skip(f"flex_attention unsupported on this backend: {exc}")
    assert y.is_complex() and y.shape == x.shape
