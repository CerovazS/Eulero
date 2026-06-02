# eulero

**Plug & play complex-valued neural network building blocks for PyTorch.**

`eulero` is a standalone library of complex-valued (and real-compatible) building
blocks - activations, attention, convolutions, normalization, positional
embeddings, patch merging - designed to be dropped into any PyTorch model. It was
extracted and refactored from the `ar_spectra` audio framework so the same blocks
can be reused across projects (complex VAEs, **source separation**, codecs, ...).

This library was built for **EuleroDec**, a complex-valued RVQ-VAE for audio
coding ([paper](https://arxiv.org/pdf/2601.17517)).

Everything lives under the `eulero.nn` namespace:

```python
import torch
from eulero.nn import CMultiHeadedAttention, ComplexRMSNorm, get_activation

x = torch.randn(2, 128, 64, dtype=torch.complex64)   # (B, L, C) complex tokens
norm = ComplexRMSNorm(64)
act  = get_activation("CReLU", is_complex=True)
y = act(norm(x))
```

---

## Installation

The package targets Python ≥ 3.10 and PyTorch ≥ 2.5.

```bash
# from a local clone
uv add /path/to/eulero          # or: pip install -e /path/to/eulero

# directly from git
uv add "eulero @ git+https://github.com/CerovazS/eulero.git"
# or
pip install "git+https://github.com/CerovazS/eulero.git"
```

> PyTorch is intentionally **not** pinned to a CUDA wheel index - install the
> torch build that matches your platform first, then add `eulero`.

---

## What's inside

All modules are importable from `eulero.nn` directly, or from their submodule for
the full catalog (e.g. `from eulero.nn.normalization.complex import ComplexBatchNorm1d`).

| Category | Submodule | Highlights |
|---|---|---|
| **Activations** | `eulero.nn.activations` | `ModReLU{,1d,2d,2dPerFreq}`, `CReLU`, `zReLU`, `magReLU`, `CardioidActivation`, `CPReLU`, `SplitReLU`, `ComplexGELU{1d,2d}`, `CSiLU`/`Abs_SiLU`, `Snake1dComplex`, `CTanh`/`CGLU`/`CMish`/`CSoftplus`/`CCeLU`, factory `get_activation(...)` |
| **Attention** | `eulero.nn.attention` | `CMultiHeadedAttention` (complex, `flex_attention` backend + SDPA fallback), real `MultiHeadedAttention`, `RelPositionMultiHeadedAttention` |
| **Convolution** | `eulero.nn.conv` | `SConv1d`/`SConv2d`, `SConvTranspose1d/2d` (causal/asymmetric padding), `NormConv*`, `NormLinear` - all take an `is_complex` flag |
| **Normalization** | `eulero.nn.normalization` | `ComplexLayerNorm`, `ComplexConvLayerNorm1d/2d`, `ComplexGroupNorm`, `ComplexBatchNorm1d`, `ComplexWeightNorm`, **`ComplexRMSNorm`** |
| **Embeddings** | `eulero.nn.embeddings` | `ComplexPositionalEncoding`, `ComplexScaledPositionalEncoding`, real positional encodings |
| **Patch merging** | `eulero.nn.complex_patch_merging` | `PatchMergingLinearComplex`, `ConvPatchDownsampleComplex`, `PatchUnmergingLinearComplex`, `ConvPatchUpsampleComplex` |
| **Layers / misc** | `eulero.nn.layers`, `.transformer`, `.rnn` | `CLinear`, `ComplexDropout`, `PositionwiseFeedForward`, `TransformerEncoder`, `SLSTM` |

A small end-to-end example (a complex 2D conv block with normalization and a
modReLU activation, the kind of stack used in complex source-separation U-Nets)
is provided in [`examples/complex_block.py`](examples/complex_block.py).

---

## Design notes

- **Complex tensors are native.** Modules operate on `torch.complex64`/`complex128`
  tensors and preserve phase where it matters (magnitude-gated activations,
  joint Re/Im normalization).
- **Real-compatible by flag.** The conv/normalization infrastructure is shared
  with the real-valued path via an `is_complex` argument, so a model can mix both.
- **Two upstream complex libraries** are used by the normalization layers:
  [`complextorch`](https://github.com/josiahwsmith10/complextorch) (LayerNorm) and
  [`complexPyTorch`](https://github.com/wavefrontshaping/complexPyTorch) (BatchNorm).

---

## Reusing in C-VAE

`eulero.nn` deliberately mirrors the original `ar_spectra.blocks` layout, so a
host project that used to import `ar_spectra.blocks.*` can migrate with a single
find-replace `ar_spectra.blocks → eulero.nn` plus adding `eulero` as a dependency.

---

## Paper

This library underpins the building blocks of **EuleroDec**, a complex-valued
RVQ-VAE for audio coding. If you use `eulero` in academic work, please cite the
paper: <https://arxiv.org/pdf/2601.17517>.

---

## License

`eulero` is released under the **PolyForm Noncommercial License 1.0.0** (see
[LICENSE](LICENSE)). Use is permitted for **noncommercial purposes only**,
including academic research, teaching, and study at educational and public
research institutions. Any commercial use requires a separate license from the
author.
