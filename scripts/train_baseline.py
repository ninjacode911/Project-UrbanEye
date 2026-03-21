"""UrbanEye baseline training script — YOLOv11 on driving dataset.

Tuned for RTX 5070 Laptop (8.5GB VRAM):
  - YOLOv11n: batch=16, imgsz=640 (~4GB VRAM)
  - YOLOv11m: batch=8, imgsz=640 (~6GB VRAM)

Usage:
  python train_baseline.py                    # default: YOLOv11n, 50 epochs
  python train_baseline.py --model m          # YOLOv11m (more accurate)
  python train_baseline.py --epochs 100       # full training run
"""

import argparse
import sys
from pathlib import Path


def get_training_config(model_size: str = "n", epochs: int = 50) -> tuple[dict, str]:
    """Get training configuration tuned for RTX 5070 Laptop.

    Args:
        model_size: 'n' for nano (fast) or 'm' for medium (accurate).
        epochs: Number of training epochs.

    Returns:
        Dict of training kwargs for model.train().
    """
    data_path = Path.home() / "urbaneye" / "data" / "urbaneye_baseline" / "dataset.yaml"

    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        print("Run prepare_dataset.py first!")
        sys.exit(1)

    # Base config (common to both model sizes)
    config = {
        "data": str(data_path),
        "epochs": epochs,
        "imgsz": 640,
        "device": 0,
        "amp": True,  # Mixed precision — saves VRAM
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,  # Final LR = lr0 * lrf (cosine schedule)
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "mosaic": 1.0,  # Mosaic augmentation (4 images/sample)
        "mixup": 0.1,
        "degrees": 5.0,  # Random rotation
        "perspective": 0.001,
        "flipud": 0.0,  # No vertical flip (cars don't fly)
        "fliplr": 0.5,
        "cls": 0.5,
        "plots": True,
        "save_period": 10,  # Checkpoint every 10 epochs
        "patience": 20,  # Early stopping patience
        "workers": 4,
        "project": str(Path.home() / "urbaneye" / "runs"),
    }

    # Model-specific config
    if model_size == "n":
        config.update(
            {
                "batch": 16,  # YOLOv11n fits batch=16 in 8.5GB
                "name": "yolov11n_baseline",
            }
        )
        model_weights = "yolo11n.pt"
    elif model_size == "m":
        config.update(
            {
                "batch": 8,  # YOLOv11m needs smaller batch for 8.5GB
                "name": "yolov11m_baseline",
            }
        )
        model_weights = "yolo11m.pt"
    else:
        print(f"ERROR: Unknown model size '{model_size}'. Use 'n' or 'm'.")
        sys.exit(1)

    return config, model_weights


def main():
    parser = argparse.ArgumentParser(description="UrbanEye YOLOv11 Baseline Training")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="n",
        choices=["n", "m"],
        help="Model size: 'n' (nano, fast) or 'm' (medium, accurate). Default: n",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=50,
        help="Number of training epochs. Default: 50",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    args = parser.parse_args()

    config, model_weights = get_training_config(args.model, args.epochs)

    print("=" * 60)
    print("UrbanEye — YOLOv11 Baseline Training")
    print("=" * 60)
    print(f"  Model:      YOLOv11{args.model} ({model_weights})")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {config['batch']}")
    print(f"  Image size: {config['imgsz']}")
    print(f"  Dataset:    {config['data']}")
    print(f"  Output:     {config['project']}/{config['name']}")
    print(f"  AMP:        {config['amp']}")
    print("=" * 60)

    # Import ultralytics here (heavy import, only when needed)
    from ultralytics import YOLO

    if args.resume:
        # Resume from last checkpoint
        last_pt = Path(config["project"]) / config["name"] / "weights" / "last.pt"
        if last_pt.exists():
            print(f"\nResuming from: {last_pt}")
            model = YOLO(str(last_pt))
        else:
            print(f"\nNo checkpoint found at {last_pt}, starting fresh.")
            model = YOLO(model_weights)
    else:
        model = YOLO(model_weights)

    print("\nStarting training...\n")

    # Train
    results = model.train(**config)

    # Results summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    print(f"  Best weights: {best_pt}")
    print(f"  Results dir:  {save_dir}")

    # Export to ONNX for deployment
    print("\nExporting to ONNX...")
    model_best = YOLO(str(best_pt))
    onnx_path = model_best.export(format="onnx", imgsz=640, half=False)
    print(f"  ONNX model:   {onnx_path}")

    print("\n" + "=" * 60)
    print("Next steps:")
    print(f"  1. Check training curves in: {save_dir}/results.png")
    print(f"  2. Run inference: yolo predict model={best_pt} source=your_video.mp4")
    print("  3. The ONNX model is ready for the HuggingFace demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
