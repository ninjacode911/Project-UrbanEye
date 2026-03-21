"""UrbanEye — Autonomous Driving Perception Pipeline Demo."""

import os
import tempfile
import time
from pathlib import Path

import cv2


# ---------------------------------------------------------------------------
# Device detection — force CPU if CUDA arch unsupported
# ---------------------------------------------------------------------------
def _detect_device() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        cap = torch.cuda.get_device_capability(0)
        supported = torch.cuda.get_arch_list()
        device_arch = f"sm_{cap[0]}{cap[1]}"
        for s in supported:
            if device_arch in s:
                torch.zeros(1, device="cuda")
                return "0"
        return "cpu"
    except Exception:
        return "cpu"


_DEVICE = _detect_device()
if _DEVICE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("[UrbanEye] Running on CPU (GPU not compatible with this PyTorch)")
else:
    print(f"[UrbanEye] Running on GPU device {_DEVICE}")


SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "models" / "best.pt"

CLASS_NAMES = ["vehicle", "pedestrian", "cyclist", "traffic_light", "traffic_sign"]
CLASS_DISPLAY = ["Vehicle", "Pedestrian", "Cyclist", "Traffic Light", "Traffic Sign"]
CLASS_COLORS = {
    "vehicle": (0, 220, 80),
    "pedestrian": (60, 60, 255),
    "cyclist": (0, 160, 255),
    "traffic_light": (0, 235, 235),
    "traffic_sign": (220, 50, 220),
}

_model = None


def get_model():
    global _model
    if _model is None and MODEL_PATH.exists():
        print(f"[UrbanEye] Loading model from {MODEL_PATH}...")
        from ultralytics import YOLO

        _model = YOLO(str(MODEL_PATH))
        # Force to correct device
        _model.to(f"cuda:{_DEVICE}" if _DEVICE != "cpu" else "cpu")
        print(f"[UrbanEye] Model loaded on {'CPU' if _DEVICE == 'cpu' else 'GPU'}")
    return _model


def draw_detections(frame, results, class_filter, conf_thresh):
    out = frame.copy()
    n = 0
    counts = {c: 0 for c in CLASS_NAMES}
    if not results or len(results) == 0:
        return out, 0, counts
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return out, 0, counts

    for i in range(len(boxes)):
        conf = float(boxes.conf[i])
        cls_id = int(boxes.cls[i])
        if conf < conf_thresh or cls_id >= len(CLASS_NAMES):
            continue
        cls = CLASS_NAMES[cls_id]
        if cls not in class_filter:
            continue

        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
        c = CLASS_COLORS.get(cls, (200, 200, 200))
        tid = f" #{int(boxes.id[i])}" if boxes.id is not None else ""

        # fill
        ov = out.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), c, -1)
        cv2.addWeighted(ov, 0.15, out, 0.85, 0, out)
        # border
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2, cv2.LINE_AA)
        # corners
        cl = min(18, (x2 - x1) // 4, (y2 - y1) // 4)
        for (cx, cy), (dx, dy) in [
            ((x1, y1), (1, 1)),
            ((x2, y1), (-1, 1)),
            ((x1, y2), (1, -1)),
            ((x2, y2), (-1, -1)),
        ]:
            cv2.line(out, (cx, cy), (cx + dx * cl, cy), c, 3, cv2.LINE_AA)
            cv2.line(out, (cx, cy), (cx, cy + dy * cl), c, 3, cv2.LINE_AA)
        # label
        lbl = f"{cls}{tid} {conf:.0%}"
        (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        ly = max(0, y1 - lh - 10)
        cv2.rectangle(out, (x1, ly), (x1 + lw + 10, y1), c, -1, cv2.LINE_AA)
        cv2.putText(
            out, lbl, (x1 + 5, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA
        )
        n += 1
        counts[cls] += 1
    return out, n, counts


def draw_hud(frame, fps, tracks, tracker, fnum, ftotal):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    fc = (0, 230, 118) if fps > 20 else (0, 200, 255) if fps > 10 else (80, 80, 255)
    cv2.putText(
        frame, f"FPS {fps:.0f}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, fc, 2, cv2.LINE_AA
    )
    tt = tracker.upper()
    (tw, _), _ = cv2.getTextSize(tt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(
        frame,
        tt,
        (w // 2 - tw // 2, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (160, 170, 200),
        1,
        cv2.LINE_AA,
    )
    tr = f"TRACKS {tracks}"
    (tw2, _), _ = cv2.getTextSize(tr, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(
        frame, tr, (w - tw2 - 14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA
    )
    if ftotal > 0:
        p = fnum / ftotal
        cv2.rectangle(frame, (0, h - 3), (int(w * p), h), (59, 130, 246), -1)
    return frame


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------
MAX_FRAMES_CPU = 150  # Cap at ~6 seconds of 25fps video on CPU


def _check_mp4_playable(path: str) -> bool:
    """Check if an mp4 file uses a browser-compatible codec."""
    try:
        import subprocess

        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        codec = r.stdout.strip()
        return codec in ("h264", "vp9", "av1")
    except Exception:
        return False


def run_video(video, tracker_type, confidence, classes):
    if video is None:
        return None, "Upload a video to begin."

    model = get_model()
    if model is None:
        return None, "Model not found. Place best.pt in hf_space/models/"

    print(f"\n[UrbanEye] Processing video: {video}")
    print(f"[UrbanEye] Tracker={tracker_type}, Conf={confidence}, Device={_DEVICE}")

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return None, f"Cannot open video: {video}"

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Cap frames on CPU to prevent hanging
    max_frames = total
    if _DEVICE == "cpu" and total > MAX_FRAMES_CPU:
        max_frames = MAX_FRAMES_CPU
        print(f"[UrbanEye] CPU mode: capping at {max_frames} frames (of {total})")

    print(f"[UrbanEye] Video: {w}x{h} @ {fps_in:.0f}fps, {total} frames, processing {max_frames}")

    # Write as raw AVI first, then always convert to H.264 mp4 via ffmpeg
    tmp = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
    raw_path = tmp.name
    tmp.close()
    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"XVID"), fps_in, (w, h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"MJPG"), fps_in, (w, h))
    print(f"[UrbanEye] Writing raw video to {raw_path}")

    tracker_yaml = "bytetrack.yaml" if tracker_type == "ByteTrack" else "botsort.yaml"
    sel = [CLASS_NAMES[CLASS_DISPLAY.index(c)] for c in (classes or CLASS_DISPLAY)]

    frame_count, total_det = 0, 0
    ct = {nm: 0 for nm in CLASS_NAMES}
    times = []

    while frame_count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.time()
        res = model.track(
            frame,
            persist=True,
            conf=confidence,
            tracker=tracker_yaml,
            verbose=False,
            device=_DEVICE,
        )
        dt = time.time() - t0
        times.append(dt)
        fps = 1 / dt if dt > 0 else 0

        ann, nt, co = draw_detections(frame, res, sel, confidence)
        ann = draw_hud(ann, fps, nt, tracker_type, frame_count, max_frames)
        writer.write(ann)

        frame_count += 1
        total_det += nt
        for k, v in co.items():
            ct[k] += v

        # Print progress every 10 frames
        if frame_count % 10 == 0 or frame_count == 1:
            print(f"[UrbanEye] Frame {frame_count}/{max_frames} | FPS: {fps:.1f} | Tracks: {nt}")

    cap.release()
    writer.release()

    # Convert AVI → H.264 MP4 for browser playback
    import subprocess

    out_path = raw_path.replace(".avi", "_out.mp4")
    print("[UrbanEye] Converting to H.264 with ffmpeg...")
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                raw_path,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                out_path,
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            print(f"[UrbanEye] ffmpeg error: {result.stderr[-200:]}")
            out_path = raw_path  # fallback to raw avi
        else:
            print(f"[UrbanEye] Converted: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
            os.unlink(raw_path)  # clean up raw avi
    except Exception as e:
        print(f"[UrbanEye] ffmpeg failed: {e}")
        out_path = raw_path

    af = 1 / (sum(times) / len(times)) if times else 0
    print(f"[UrbanEye] Done! {frame_count} frames, avg {af:.1f} FPS, {total_det} detections\n")

    m = f"Tracker      {tracker_type}\n"
    m += f"Frames       {frame_count}"
    if max_frames < total:
        m += f" (capped from {total})"
    m += f"\nAvg FPS      {af:.1f}\n"
    m += f"Device       {'CPU' if _DEVICE == 'cpu' else 'GPU'}\n"
    m += f"Detections   {total_det}\n"
    m += f"Threshold    {confidence}\n\n"
    for nm in CLASS_NAMES:
        dn = CLASS_DISPLAY[CLASS_NAMES.index(nm)]
        bar = "\u2588" * min(20, ct[nm] // max(1, frame_count))
        m += f"  {dn:15s} {ct[nm]:>5d}  {bar}\n"
    return out_path, m


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------
def run_image(image, confidence, classes):
    if image is None:
        return None, "Upload an image to detect."

    model = get_model()
    if model is None:
        return None, "Model not found."

    sel = [CLASS_NAMES[CLASS_DISPLAY.index(c)] for c in (classes or CLASS_DISPLAY)]

    print(f"[UrbanEye] Running image detection, conf={confidence}, device={_DEVICE}")
    t0 = time.time()
    res = model(image, conf=confidence, verbose=False, device=_DEVICE)
    dt = time.time() - t0
    print(f"[UrbanEye] Image inference: {dt * 1000:.0f}ms")

    ann, nd, co = draw_detections(image, res, sel, confidence)
    m = f"Detections   {nd}\nInference    {dt * 1000:.1f} ms\nDevice       {'CPU' if _DEVICE == 'cpu' else 'GPU'}\n\n"
    for nm in CLASS_NAMES:
        if co[nm] > 0:
            m += f"  {CLASS_DISPLAY[CLASS_NAMES.index(nm)]}: {co[nm]}\n"
    return ann, m


# ---------------------------------------------------------------------------
# Theme + CSS
# ---------------------------------------------------------------------------
THEME_KWARGS = {
    "body_background_fill": "#0b0f19",
    "body_background_fill_dark": "#0b0f19",
    "block_background_fill": "#111827",
    "block_background_fill_dark": "#111827",
    "block_border_color": "#1e293b",
    "block_border_color_dark": "#1e293b",
    "block_label_background_fill": "#1e293b",
    "block_label_background_fill_dark": "#1e293b",
    "block_label_text_color": "#94a3b8",
    "block_label_text_color_dark": "#94a3b8",
    "block_title_text_color": "#e2e8f0",
    "block_title_text_color_dark": "#e2e8f0",
    "body_text_color": "#e2e8f0",
    "body_text_color_dark": "#e2e8f0",
    "body_text_color_subdued": "#64748b",
    "body_text_color_subdued_dark": "#64748b",
    "button_primary_background_fill": "linear-gradient(135deg, #3b82f6, #6366f1)",
    "button_primary_background_fill_dark": "linear-gradient(135deg, #3b82f6, #6366f1)",
    "button_primary_background_fill_hover": "linear-gradient(135deg, #2563eb, #4f46e5)",
    "button_primary_background_fill_hover_dark": "linear-gradient(135deg, #2563eb, #4f46e5)",
    "button_primary_text_color": "#ffffff",
    "button_primary_text_color_dark": "#ffffff",
    "input_background_fill": "#0f172a",
    "input_background_fill_dark": "#0f172a",
    "input_border_color": "#1e293b",
    "input_border_color_dark": "#1e293b",
    "input_border_color_focus": "#3b82f6",
    "input_border_color_focus_dark": "#3b82f6",
    "border_color_primary": "#1e293b",
    "border_color_primary_dark": "#1e293b",
}

CSS = """
* { font-family: 'Inter', sans-serif !important; }
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; padding: 0 2rem !important; }

/* Radio & Checkbox label fix */
label.svelte-19qdtil,
.svelte-19qdtil[class*="selected"],
fieldset label {
    background: #1e293b !important;
    color: #cbd5e1 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}
label.svelte-19qdtil:hover, fieldset label:hover {
    background: #334155 !important;
    color: #f1f5f9 !important;
    border-color: #3b82f6 !important;
}
label.svelte-19qdtil.selected, .svelte-19qdtil.selected {
    background: rgba(59, 130, 246, 0.15) !important;
    color: #60a5fa !important;
    border-color: #3b82f6 !important;
}
label.svelte-19qdtil span, fieldset label span,
input[type="radio"] + span, input[type="checkbox"] + span {
    color: inherit !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    text-transform: none !important;
    letter-spacing: normal !important;
}
input[type="checkbox"] { accent-color: #3b82f6 !important; }

/* hero */
#hero-title { text-align: center; margin-bottom: -8px; }
#hero-title h1 {
    font-size: 2.6rem; font-weight: 800; letter-spacing: -0.04em;
    background: linear-gradient(135deg, #60a5fa, #34d399, #a78bfa);
    background-size: 200% 200%;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: glow 8s ease infinite;
}
@keyframes glow {
    0%,100%{background-position:0% 50%} 50%{background-position:100% 50%}
}
#hero-sub p { text-align:center; color:#94a3b8; font-size:0.95rem; margin-top:0; }
#hero-classes p {
    text-align:center; font-size:0.82rem; color:#64748b;
    padding:8px 0 4px 0; border-top:1px solid #1e293b; margin-top:4px;
}

/* tabs */
.tab-nav { border-bottom: 1px solid #1e293b !important; }
.tab-nav button {
    font-weight: 600 !important; font-size: 0.88rem !important;
    color: #64748b !important; padding: 10px 20px !important;
    border: none !important; border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
.tab-nav button:hover { color: #e2e8f0 !important; }
.tab-nav button.selected {
    color: #60a5fa !important;
    border-bottom-color: #3b82f6 !important;
}

/* button */
button.primary, button.lg {
    font-weight: 700 !important; text-transform: uppercase !important;
    letter-spacing: 0.05em !important; font-size: 0.88rem !important;
    border-radius: 10px !important; padding: 11px 0 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 14px rgba(59,130,246,0.25) !important;
}
button.primary:hover, button.lg:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.35) !important;
}

/* metrics mono */
#metrics textarea, #img-metrics textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important; line-height: 1.65 !important;
    color: #6ee7b7 !important;
    background: #0a0f1a !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
}

/* labels */
label span { font-size: 0.75rem !important; letter-spacing:0.07em !important; text-transform:uppercase !important; }

/* about code blocks */
.prose pre {
    background: #0f172a !important; border: 1px solid #1e293b !important;
    border-radius: 10px !important; font-family: 'JetBrains Mono', monospace !important;
}
.prose code { color: #60a5fa !important; background: #1e293b !important; border-radius: 4px !important; padding: 1px 6px !important; }
.prose table th { background: #1e293b !important; color: #60a5fa !important; }
.prose table td { border-color: #1e293b !important; }

/* entrance */
.block { animation: fadeUp 0.5s ease both; }
@keyframes fadeUp { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }

footer { opacity: 0.3 !important; }
"""

JS = """
() => {
    document.querySelectorAll('.block').forEach((el,i) => {
        el.style.animationDelay = `${i*60}ms`;
    });
}
"""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main():
    import gradio as gr

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ).set(**THEME_KWARGS)

    with gr.Blocks(theme=theme, css=CSS, js=JS, title="UrbanEye") as demo:
        gr.Markdown("# UrbanEye", elem_id="hero-title")
        gr.Markdown(
            "Autonomous Driving Perception &nbsp;&bull;&nbsp; YOLOv11 &nbsp;&bull;&nbsp; "
            "ByteTrack + DeepSORT",
            elem_id="hero-sub",
        )
        gr.Markdown(
            "<span style='color:#34d399'>Vehicle</span> &nbsp;&bull;&nbsp; "
            "<span style='color:#f87171'>Pedestrian</span> &nbsp;&bull;&nbsp; "
            "<span style='color:#fb923c'>Cyclist</span> &nbsp;&bull;&nbsp; "
            "<span style='color:#facc15'>Traffic Light</span> &nbsp;&bull;&nbsp; "
            "<span style='color:#c084fc'>Traffic Sign</span>",
            elem_id="hero-classes",
        )

        with gr.Tabs():
            with gr.TabItem("Video Tracking"):
                with gr.Row():
                    with gr.Column(scale=5):
                        vid_in = gr.Video(label="Input Video", height=340)
                    with gr.Column(scale=5):
                        vid_out = gr.Video(label="Output", height=340)

                with gr.Row():
                    with gr.Column(scale=3):
                        tracker = gr.Radio(
                            ["ByteTrack", "DeepSORT"], value="ByteTrack", label="Tracker"
                        )
                    with gr.Column(scale=3):
                        conf = gr.Slider(0.10, 0.90, 0.35, step=0.05, label="Confidence")
                    with gr.Column(scale=4):
                        cls = gr.CheckboxGroup(CLASS_DISPLAY, value=CLASS_DISPLAY, label="Classes")

                run_btn = gr.Button("Run Detection + Tracking", variant="primary", size="lg")
                met = gr.Textbox(label="Metrics", lines=9, interactive=False, elem_id="metrics")
                run_btn.click(run_video, [vid_in, tracker, conf, cls], [vid_out, met])

            with gr.TabItem("Image Detection"):
                with gr.Row():
                    with gr.Column(scale=5):
                        img_in = gr.Image(label="Input Image", type="numpy", height=360)
                    with gr.Column(scale=5):
                        img_out = gr.Image(label="Result", height=360)

                with gr.Row():
                    with gr.Column(scale=4):
                        img_conf = gr.Slider(0.10, 0.90, 0.35, step=0.05, label="Confidence")
                    with gr.Column(scale=6):
                        img_cls = gr.CheckboxGroup(
                            CLASS_DISPLAY, value=CLASS_DISPLAY, label="Classes"
                        )

                img_btn = gr.Button("Detect Objects", variant="primary", size="lg")
                img_met = gr.Textbox(
                    label="Results", lines=6, interactive=False, elem_id="img-metrics"
                )
                img_btn.click(run_image, [img_in, img_conf, img_cls], [img_out, img_met])

            with gr.TabItem("About"):
                gr.Markdown("""
## Architecture

```
CARLA Simulator ──▶ YOLOv11n ──▶ ByteTrack / DeepSORT ──▶ MOT Evaluation
   50K+ frames      5 classes     Dual tracker              MOTA · IDF1
```

| Spec | Detail |
|------|--------|
| Model | YOLOv11n — 2.6M params, 6.4 GFLOPs |
| Training | 50 epochs, AdamW, cosine LR, AMP |
| GPU | RTX 5070 Laptop (8.5 GB) |
| mAP@50 | 0.47 (baseline) |
| Classes | Vehicle, Pedestrian, Cyclist, Traffic Light, Traffic Sign |

| Tracker | Method | FPS | Strength |
|---------|--------|-----|----------|
| ByteTrack | IoU + Kalman, two-stage matching | 60+ | Speed, low-conf rescue |
| DeepSORT | IoU + Kalman + MobileNetV2 Re-ID | 25-35 | Identity through occlusion |

---
*Navnit Amrutharaj &bull; [github.com/ninjacode911](https://github.com/ninjacode911)*
                """)

    print("[UrbanEye] Starting server...")
    demo.launch(share=False)


if __name__ == "__main__":
    main()
