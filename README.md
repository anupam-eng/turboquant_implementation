# TurboQuant + RotorQuant Implementation

A from-scratch PyTorch implementation of [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026, Google Research) for **KV cache compression** in LLM inference, **plus the full RotorQuant family** (PlanarQuant, IsoQuant, RaBitQ, post-prefill cache, fused Triton kernels) layered on top.

## Features

- **Two quantizer families**:
  - *TurboQuant* — dense `d × d` random rotation (the original baseline)
  - *RotorQuant* — block-diagonal rotations: PlanarQuant (Givens 2D) and IsoQuant (quaternion 4D), plus RaBitQ (1-bit)
- **Post-prefill quantization** (`DeferredQuantCache`) — avoids the layer-compounding catastrophe (PPL > 1000 → coherent text)
- **Symmetric K + V quantization** with construction-time inverse-rotation safety check (the canonical `PPL=15K` bug, caught at init)
- **Asymmetric K/V bit allocation** — e.g., Keys at 4-bit, Values at 2-bit
- **Triton kernels** with PyTorch fallback so the codebase runs on Apple Silicon (no CUDA needed for development)
- **Fused quantize+attention kernel** — quantized K never touches VRAM; arithmetic intensity 0.5 → 500 FLOPs/byte
- **GQA support** — Llama 3 / Qwen 2.5 / Mistral grouped-query attention
- **Qwen 3.5 hybrid architecture** — linear-attention + full-attention layers handled correctly

## Quick Start

### Existing TurboQuant (one-liner)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant.attention_patch import enable_turboquant
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-0.8B", dtype=torch.float32, trust_remote_code=True
).to("mps")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)

enable_turboquant(model, key_bits=4, value_bits=2, residual_window=64)
outputs = model.generate(**tokenizer("Hello!", return_tensors="pt").to("mps"),
                         max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### New RotorQuant (production winner: IsoQuant K=4/V=4)

```python
from turboquant.rotorquant_kv_cache import enable_rotorquant

# Drop-in replacement that uses Phase 3+4+5+6 (block-diagonal rotation,
# post-prefill state machine, symmetric V with inverse-rotation safety)
enable_rotorquant(model, key_bits=4, value_bits=4, quantizer_kind="iso")
```

## Results

### MSE Distortion (10K random unit vectors, d=128)

| Bits | Empirical MSE | Ratio to Lower Bound | Paper Guarantee | Status |
|------|--------------|---------------------|-----------------|--------|
| 1 | 0.3611 | 1.444× | ≤2.7× | ✅ |
| 2 | 0.1161 | 1.857× | ≤2.7× | ✅ |
| 3 | 0.0340 | 2.173× | ≤2.7× | ✅ |
| 4 | 0.0093 | 2.386× | ≤2.7× | ✅ |

### Qwen 3.5-0.8B end-to-end on Apple M4 (greedy decoding, 120 tokens)

| Config | tok/s | full-attn KV | compression | output |
|---|---:|---:|---:|---|
| Baseline (FP32) | 7.98 | 3,408 KiB | 1.00× | coherent |
| TurboQuant K=4/V=2 (existing) | 8.49 | 991 KiB | 3.44× | coherent |
| RotorQuant Planar K=3/V=3 (Phase 3+5+6) | 7.91 | 974 KiB | 3.50× | coherent |
| **RotorQuant Iso K=4/V=4 (Phase 4+5+6)** | **8.82** | **491 KiB** | **6.95×** | coherent |

The new **RotorQuant IsoQuant K=4/V=4** wins on both compression *and* speed — 6.95× compression on full-attention KV and 10.5% faster than FP32 baseline. Reproduce with `examples/benchmark_rotorquant.py`.

## RotorQuant Phase Status

| # | Phase | Status | Module | Headline |
|---|---|:---:|---|---|
| 1 | Norm separation | ✅ pre-existing | `quantizer.py` | 4-bit MSE within 0.1% theoretical |
| 2 | QJL Stage 2 | ✅ verified | `quantizer.py` | slope 0.9991 vs MSE-only's 0.9671 |
| 3 | PlanarQuant | ✅ implemented | `planarquant.py` | 64× fewer FMAs than TurboQuant, same MSE |
| 4 | IsoQuant | ✅ implemented | `isoquant.py` | quaternion 4D blocks, MSE within 0.6% of PlanarQuant |
| 5 | DeferredQuantCache | ✅ implemented | `deferred_cache.py` | 2.21× compounding penalty avoided |
| 6 | SymmetricKVCache | ✅ implemented | `symmetric_cache.py` | attention cos sim 0.9949 correct vs -0.02 buggy |
| 7 | Triton kernels | ✅ implemented | `kernels/triton_*.py` | bit-exact parity, CUDA-ready |
| 8+9 | Fused attention + GQA | ✅ implemented | `kernels/fused_planar_attn.py` | rel L2 diff 1.7e-7; GQA scales with H_kv |
| 10 | RaBitQ (1-bit) | ✅ implemented | `rabitq.py` | 12.8× compression, slope 1.0077 with π/2 correction |
| 11 | llama.cpp integration | 📋 plan-of-record | `docs/ROTORQUANT.md` | 835 LOC plan with file map |

## Documentation

- **[docs/ROTORQUANT.md](docs/ROTORQUANT.md)** — single combined reference (1,660 lines): math, algorithms, tests, pitfalls, problems-solved-per-phase, and a per-model config decision matrix.
- **[WALKTHROUGH.md](WALKTHROUGH.md)** — original TurboQuant Stage 1 implementation walkthrough.
- **Per-phase docs** in `docs/phase{2..11}_*.md`.

## Run all tests (Apple Silicon-compatible)

```bash
PYTHONPATH=. python tests/test_qjl_unbiased.py
PYTHONPATH=. python tests/test_planarquant.py
PYTHONPATH=. python tests/test_isoquant.py
PYTHONPATH=. python tests/test_deferred_cache.py
PYTHONPATH=. python tests/test_symmetric_cache.py
PYTHONPATH=. python tests/test_rabitq.py
PYTHONPATH=. python tests/test_kernel_parity.py
PYTHONPATH=. python tests/test_fused_attention.py
PYTHONPATH=. python turboquant/verify_roundtrip.py
```

**9/9 test files pass on Apple Silicon.** On CUDA, the same tests pass and the Triton kernels run instead of the PyTorch fallback (expect 100–650× speedup on `test_kernel_parity` and 1.1–4.5× on `test_fused_attention`).

## Reproduce the end-to-end benchmark

```bash
PYTHONPATH=. python examples/benchmark_rotorquant.py    # 4-config table
PYTHONPATH=. python examples/cache_breakdown.py         # per-layer-type bytes
```

~60 seconds total on Apple M4 / MPS. See `docs/ROTORQUANT.md` § *End-to-End Validation Details* for full numbers and output excerpts.

## Install

```bash
pip install -r requirements.txt
```

## License

MIT
