# GlioODE / GlioForecast

A research codebase that grew across five iterations:

1. **GlioODE (iter-1/2/3):** a 3D joint-diffusion segmentation/generation model for brain tumours. A Vision Transformer trunk whose discrete block stack is replaced by a Neural ODE (`torchdiffeq`), UNETR-style conv decoder, axial RoPE + QK rms-norm + Flash/SDPA attention.
2. **GlioForecast (iter-4):** a Bayesian inverse model for glioma growth on top of GlioODE. The encoder predicts logits for Fisher-Kolmogorov parameters `(D(x), ρ(x), seed_map(x))`; a differentiable FK Neural-ODE decoder (`dc/ds = ∇·(D∇c) + ρc(1-c) + f_θ`) integrates forward in physical time. Trained with DDPM ε-loss + FK reconstruction loss.
3. **Validation (iter-5):** synthetic ground-truth metrics (`D_mse`, `rho_mse`, `seed_xyz_dist`, `c_forecast_dice@0.1`), per-tissue breakdowns, TensorBoard image summaries, best-by-Dice checkpoint selection.

## Hardware

- Designed for a **single NVIDIA A100 (40 GB or 80 GB)**.
- bf16 mixed precision (the Neural ODE solvers stay fp32 internally for stability).
- No DDP / multi-GPU support yet.

## Dataset
- Download the PredictGBM dataset below:
  https://huggingface.co/datasets/LZimmer/PREDICT-GBM/tree/main
  We extract **T1Gd,**, **CSF** (cerebrospinal fluid mask), **Wm** (White Matter Mask), **Gm** (Gray Matter Mask) from this dataset.
- Future work will revolve around synthetically generating data from a VTU Mesh Geometry

## Install

```bash
# 1. Python 3.11+ environment.
pip install -e .
pip install -r requirements.txt

# 2. (Optional but recommended on A100) FlashAttention 2:
pip install flash-attn --no-build-isolation
```

If `flash-attn` isn't installed, the model falls back to PyTorch's SDPA — slower but functionally identical.


Hydra overrides:

```bash
python train_forecast.py training.batch_size=1 training.amp=true training.total_iters=100000
```

## Run inference (single case)

```bash
python forecast.py infer \
  --ckpt=ckpt/glio_forecast/best.pt \
  --case=/data/gliomasolver/case_042 \
  --horizon=180 \
  --output_dir=out/case_042/
```

Outputs:

- `out/case_042/c_forecast.nii.gz` — predicted tumour density at `horizon` days.
- `out/case_042/D.nii.gz`, `rho.nii.gz`, `seed_map.nii.gz` — inferred FK parameter fields.

## Tests

```bash
pytest -q
# Expected: 97 passed.
```
