# Phase 13: HuggingFace Spaces Deployment

**Status:** Completed
**Date:** 2026-03-21
**URL:** https://huggingface.co/spaces/NinjainPJs/UrbanEye

---

## Objective

Deploy the UrbanEye demo to HuggingFace Spaces as a publicly accessible Gradio application with real YOLOv11 inference and ByteTrack tracking.

---

## Pre-Deployment Checklist Results

| Check | Result |
|-------|--------|
| Full test suite | 270/270 passed |
| Ruff lint | Clean (0 errors) |
| Ruff format | Clean (54 files) |
| No `.env` files | PASS |
| No API keys/tokens in code | PASS |
| No hardcoded IPs/paths | PASS |
| `.gitignore` covers secrets | PASS |
| PDF excluded from repo | PASS |
| Model weights in `.gitignore` | PASS (main repo) |

---

## Deployment Details

### HuggingFace Space Configuration

| Setting | Value |
|---------|-------|
| Space ID | `NinjainPJs/UrbanEye` |
| SDK | Gradio |
| Python | 3.10 |
| Hardware | CPU (free tier) |
| License | Apache 2.0 |

### Files Uploaded

| File | Size | Purpose |
|------|------|---------|
| `app.py` | 22 KB | Self-contained Gradio app with YOLOv11 + ByteTrack |
| `models/best.pt` | 5.3 MB | Trained YOLOv11n weights |
| `models/best.onnx` | 11 MB | ONNX export (for future CPU optimization) |
| `requirements.txt` | — | Python dependencies |
| `packages.txt` | — | System packages (ffmpeg, libgl1) |
| `README.md` | — | HF Space metadata (SDK, emoji, license) |

### Key Technical Decisions for Deployment

1. **Auto device detection** — `_detect_device()` checks CUDA availability and architecture compatibility. Falls back to CPU gracefully on HF Spaces (no GPU) and on Windows (unsupported sm_120).

2. **XVID → H.264 ffmpeg conversion** — OpenCV's `VideoWriter` with `mp4v` or `avc1` codec doesn't produce browser-playable MP4s. Solution: write as XVID AVI, then convert to H.264 MP4 using ffmpeg with `libx264`, `yuv420p`, and `+faststart` flags.

3. **CPU frame cap** — On CPU, processing is capped at 150 frames (`MAX_FRAMES_CPU`) to prevent the request from timing out on HF Spaces free tier (300s timeout).

4. **Model included in Space** — The 5.3MB `best.pt` is small enough to include directly in the Space repo. Larger models would need HF Model Hub hosting with `hf_hub_download()`.

---

## Bugs Fixed During Deployment

1. **`_model.to('0')` → `RuntimeError: Invalid device string`** — PyTorch requires `"cuda:0"`, not `"0"`. Fixed to `f"cuda:{_DEVICE}"`.

2. **Invisible radio/checkbox labels** — Gradio theme set label background and text to the same color (`rgb(226, 232, 240)`). Fixed with explicit CSS overrides for `.svelte-19qdtil` labels.

3. **White output panel** — `cv2.VideoWriter` with `mp4v` codec produces MP4 files that browsers refuse to play. Fixed by writing XVID AVI then converting to H.264 via ffmpeg subprocess.

4. **PermissionError on Windows temp files** — Gradio caches uploaded files in `%TEMP%\gradio\`. MOV files caused permission locks. Fixed by clearing cache and using MP4 format.

5. **Windows PyTorch CUDA incompatibility** — RTX 5070 Laptop (sm_120) not supported by Windows PyTorch (max sm_90). Fixed with `_detect_device()` that falls back to CPU. WSL2 PyTorch 2.10+cu128 supports sm_120 properly.

---

## Runtime Performance

| Platform | Device | FPS | Notes |
|----------|--------|-----|-------|
| WSL2 (RTX 5070) | GPU | 21-23 | Full speed, production inference |
| Windows (RTX 5070) | CPU | 2-3 | PyTorch CUDA incompatible |
| HuggingFace Spaces | CPU | ~2-5 | Free tier, capped at 150 frames |

---

## Model Upgrade: YOLOv11n → YOLOv11m (2026-03-22)

Retrained with YOLOv11m (20M params, 68.2 GFLOPs) for 100 epochs.

| Metric | YOLOv11n (old) | YOLOv11m (new) | Improvement |
|--------|---------------|---------------|-------------|
| mAP@50 | 0.473 | **0.540** | +14.2% |
| mAP@50-95 | 0.290 | **0.341** | +17.6% |
| Parameters | 2.6M | 20M | 8x |
| Training | 42 min | 3.0 hrs | 100 epochs |

Per-class AP@50: Pedestrian 0.736, Traffic Sign 0.636, Vehicle 0.592, Traffic Light 0.418, Cyclist 0.320.

Pushed to HuggingFace Space, replacing the old YOLOv11n model.
