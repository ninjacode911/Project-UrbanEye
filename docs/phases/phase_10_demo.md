# Phase 10: HuggingFace Spaces Demo

**Status:** Completed
**Date:** 2026-03-21
**Tests:** 6 new tests (263 cumulative, all passed)

---

## Objective

Build the Gradio-based demo application for HuggingFace Spaces deployment — video upload, tracker selection, confidence slider, class filtering, and annotated output with metrics.

---

## What Was Built

### 1. `urbaneye/demo/app.py` — Development Demo

Full pipeline: video → detect → track → annotate → write output. Key components:

- **`annotate_frame()`** — Draws colored bboxes, track IDs, confidence scores, FPS overlay, track count
- **`process_video()`** — Opens video with OpenCV, runs DualTracker per frame, writes annotated output
- **`ProcessingResult`** — Dataclass with total_frames, total_detections, class_counts, avg_fps

### 2. `hf_space/app.py` — Self-Contained HF Spaces Version

Completely self-contained (no `urbaneye` package imports). Gradio interface with:
- Video upload + output panels
- Tracker radio: ByteTrack / DeepSORT
- Confidence slider (0.1-0.9)
- Class checkboxes (5 classes)
- Metrics textbox
- Run button

Currently a skeleton — requires `best.onnx` model weights to be uploaded to `hf_space/models/` for live inference.

### 3. `hf_space/README.md` + `requirements.txt` — HF Spaces Config

Metadata YAML for Gradio SDK deployment and pinned dependencies.

---

## Files Created

```
urbaneye/demo/__init__.py, app.py       # Development demo
hf_space/app.py                         # Self-contained HF Spaces demo
hf_space/README.md                      # HF Spaces metadata
hf_space/requirements.txt               # Deployment dependencies
tests/integration/test_demo_app.py      # 6 integration tests
```

---

## Key Design: Two App Versions

1. **Development** (`urbaneye/demo/app.py`): Imports from the package, uses DualTracker directly
2. **Deployment** (`hf_space/app.py`): Self-contained, uses ONNX Runtime for CPU inference

This separation ensures HF Spaces can run `pip install -r requirements.txt` + `python app.py` without installing the full urbaneye package.
