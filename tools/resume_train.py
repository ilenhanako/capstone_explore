#!/usr/bin/env python3
"""
1) OOM mitigation: Auto batch, disk caching
2) 

Fresh training with auto-batch (up to 16), checkpointing, disk cache:
    python tools/resume_train.py \
        --dataset ./preprocessedDataset \
        --weights ./yolov11l.pt \
        --mode fresh --epochs 80 --imgsz 640 \
        --max-batch-test 16 --desired-eff-batch 16 --cache disk --workers 2

Resume (no auto-batch):
    python tools/resume_train.py \
        --dataset ./preprocessedDataset \
        --project_root ./yolo_training_runs \
        --project_name yolov11l_road_defects \
        --mode resume --no-autobatch --workers 2
"""
from pathlib import Path
import argparse
import sys
import yaml
import os
import gc
import traceback
import types
import torch
from ultralytics import YOLO

DEFAULT_TAXONOMY = {
    0: "road_crack_longitudinal",
    1: "road_crack_transverse",
    2: "road_crack_alligator",
    3: "road_rutting",
    4: "pothole",
    5: "marking_faded",
    6: "distractor_manhole",
    7: "distractor_patch",
}

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def ensure_split_structure(dataset_path: Path):
    missing = []
    for split in ("train", "val", "test"):
        img_dir = dataset_path / "images" / split
        lbl_dir = dataset_path / "labels" / split
        if not img_dir.is_dir() or not any(img_dir.glob("*")):
            missing.append(str(img_dir))
        if not lbl_dir.is_dir() or not any(lbl_dir.glob("*.txt")):
            missing.append(str(lbl_dir))
    return missing

def fix_or_create_data_yaml(dataset_path: Path, taxonomy: dict = None):
    taxonomy = taxonomy or DEFAULT_TAXONOMY
    data_yaml_path = dataset_path / "data.yaml"
    data = {}
    if data_yaml_path.exists():
        try:
            with open(data_yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[warn] existing data.yaml parse failed: {e}; overwriting.")
            data = {}

    data["path"] = str(dataset_path.resolve())
    data["train"] = data.get("train", "images/train")
    data["val"] = data.get("val", "images/val")
    data["test"] = data.get("test", "images/test")

    if "names" not in data:
        data["names"] = {int(k): v for k, v in taxonomy.items()}
        data["nc"] = len(data["names"])
    else:
        names = data["names"]
        if isinstance(names, dict):
            data["nc"] = len(names)
        else:
            data["names"] = {i: n for i, n in enumerate(names)}
            data["nc"] = len(data["names"])

    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"[ok] data.yaml written: {data_yaml_path}")
    return str(data_yaml_path)

def locate_checkpoint(project_root: Path, project_name: str):
    weights_dir = project_root / project_name / "weights"
    last = weights_dir / "last.pt"
    best = weights_dir / "best.pt"
    if last.exists():
        return str(last)
    if best.exists():
        return str(best)
    return None

# ---------------------------------------------------------------------------
# Memory & auto-batch probing
# ---------------------------------------------------------------------------
def _clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _synthetic_loss(out, dev):
    if isinstance(out, torch.Tensor):
        if out.requires_grad:
            return out.abs().mean()
        return torch.zeros((), device=dev, requires_grad=True)
    if isinstance(out, (list, tuple)):
        grads = [t for t in out if isinstance(t, torch.Tensor) and t.requires_grad]
        if grads:
            return sum([t.abs().mean() for t in grads])
        return torch.zeros((), device=dev, requires_grad=True)
    return torch.zeros((), device=dev, requires_grad=True)

def try_batch_size(model, imgsz=640, batch=4, device='cuda', use_amp=True):
    _clear_cuda()
    dev = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    mm = getattr(model, "model", model)
    mm.to(dev).train()
    opt = torch.optim.SGD(mm.parameters(), lr=1e-6)

    try:
        scaler = torch.amp.GradScaler("cuda") if (use_amp and dev.type == 'cuda') else None
        imgs = torch.randn(batch, 3, imgsz, imgsz, device=dev, dtype=torch.float32)
        opt.zero_grad(set_to_none=True)
        if scaler:
            with torch.amp.autocast("cuda"):
                out = mm(imgs)
                loss = _synthetic_loss(out, dev)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            out = mm(imgs)
            loss = _synthetic_loss(out, dev)
            loss.backward()
            opt.step()
        peak = torch.cuda.max_memory_allocated(dev) if dev.type == 'cuda' else 0
        _clear_cuda()
        return True, int(peak)
    except RuntimeError as e:
        _clear_cuda()
        msg = str(e)
        if "out of memory" in msg.lower():
            return False, "OOM"
        return False, msg
    except Exception as e:
        _clear_cuda()
        return False, str(e)

def find_max_batch(model, imgsz=640, device='cuda', use_amp=True, max_batch=16, min_batch=1, verbose=True):
    lo, hi = min_batch, max_batch
    results = {}
    ok, info = try_batch_size(model, imgsz, hi, device, use_amp)
    results[hi] = (ok, info)
    if ok:
        while ok and hi < 64:
            lo = hi
            hi = min(hi * 2, 64)
            ok, info = try_batch_size(model, imgsz, hi, device, use_amp)
            results[hi] = (ok, info)
            if verbose:
                print(f"[probe] batch {hi}: ok={ok}, info={info}")
    else:
        if verbose:
            print(f"[probe] upper bound {max_batch} failed: {info}")

    if not results.get(lo, (None,))[0]:
        found = False
        for b in range(lo, 0, -1):
            ok, info = try_batch_size(model, imgsz, b, device, use_amp)
            results[b] = (ok, info)
            if ok:
                lo = b
                found = True
                break
        if not found:
            return 0, results

    if hi <= lo:
        hi = lo * 2

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if mid not in results:
            ok, info = try_batch_size(model, imgsz, mid, device, use_amp)
            results[mid] = (ok, info)
        else:
            ok, info = results[mid]
        if verbose:
            print(f"[probe] try {mid}: ok={ok}")
        if ok:
            lo = mid
        else:
            hi = mid

    if verbose:
        print(f"[autobatch] max per-step batch fitting: {lo}")
    return lo, results

# ---------------------------------------------------------------------------
# Safe gradient checkpointing (select modules only)
# ---------------------------------------------------------------------------
def enable_safe_checkpointing(model, verbose=True):
    """
    Wrap forward of selected heavy modules inside model.model.model (Sequential).
    Does NOT alter top-level indexing required by Ultralytics (Detect at -1).
    """
    det = getattr(model, "model", None)
    if det is None:
        if verbose:
            print("[ckpt] No detection model found; skipping checkpointing.")
        return {"applied": False, "modules": []}
    seq = getattr(det, "model", None)
    if not isinstance(seq, torch.nn.Sequential):
        if verbose:
            print("[ckpt] detection inner 'model' is not Sequential; skipping.")
        return {"applied": False, "modules": []}

    target_names = {"C3k2", "C2PSA", "SPPF"}
    wrapped = []

    def wrap_forward(m):
        if hasattr(m, "_orig_forward"):
            return
        orig = m.forward
        def _forward(*inp, **kw):
            return torch.utils.checkpoint.checkpoint(lambda *x: orig(*x, **kw), *inp)
        m._orig_forward = orig
        m.forward = types.MethodType(_forward, m)

    for i, layer in enumerate(seq):
        lname = layer.__class__.__name__
        if lname in target_names:
            try:
                wrap_forward(layer)
                wrapped.append(f"{i}:{lname}")
            except Exception as e:
                if verbose:
                    print(f"[ckpt] Failed wrapping {i}:{lname}: {e}")

    if verbose:
        if wrapped:
            print(f"[ckpt] Applied gradient checkpointing to: {wrapped}")
        else:
            print("[ckpt] No target modules found for checkpointing.")
    return {"applied": bool(wrapped), "modules": wrapped}

# ---------------------------------------------------------------------------
# Training workflow
# ---------------------------------------------------------------------------
def train_with_yolo(weights: str,
                    data_yaml: str,
                    project_root: str,
                    project_name: str,
                    epochs: int,
                    batch: int,
                    imgsz: int,
                    resume_ul: bool,
                    cache_option: str = "disk",
                    max_batch_test: int = 16,
                    desired_eff_batch: int = 16,
                    workers: int = 2,
                    enable_autobatch: bool = True,
                    enable_checkpointing: bool = True,
                    safe_low_batch: int = 4):
    print(f"[info] Loading model from: {weights}")
    model = YOLO(weights)

    # Auto-batch probe first
    chosen_batch = batch
    if enable_autobatch:
        try:
            print(f"[autobatch] Probing up to batch {max_batch_test}...")
            best_batch, _ = find_max_batch(
                model,
                imgsz=imgsz,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                use_amp=True,
                max_batch=max_batch_test,
                verbose=True
            )
            if best_batch <= 0:
                print(f"[autobatch] Probe failed; falling back to {safe_low_batch}")
                best_batch = safe_low_batch
            chosen_batch = min(best_batch, batch) if batch > 0 else best_batch
        except Exception as e:
            print(f"[warn] Auto-batch probing error: {e}; using requested batch={batch}")
            chosen_batch = batch
    effective_batch_concept = desired_eff_batch if desired_eff_batch > 0 else chosen_batch

    # Safe gradient checkpointing AFTER probing
    if enable_checkpointing:
        ckinfo = enable_safe_checkpointing(model, verbose=True)
    else:
        ckinfo = {"applied": False, "modules": []}

    # Prepare train args
    train_args = dict(
        data=data_yaml,
        imgsz=imgsz,
        epochs=epochs,
        batch=chosen_batch,
        workers=workers,
        device=0 if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else 'cpu',
        lr0=3e-4,
        lrf=0.01,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        project=project_root,
        name=project_name,
        exist_ok=True,
        save=True,
        seed=42,
        cos_lr=True,
        close_mosaic=10,
    )
    if resume_ul:
        train_args["resume"] = True

    c = (cache_option or "none").lower()
    if c in ("disk", "cache_disk", "cache-disk"):
        train_args["cache"] = "disk"
    elif c in ("ram", "memory", "cache"):
        train_args["cache"] = True

    print("[info] Train args:")
    for k, v in train_args.items():
        print(f"  {k}: {v}")
    print(f"[info] Conceptual effective batch target: {effective_batch_concept} (native accumulate not supported in 8.3.220)")

    # Start training
    try:
        print("[info] Starting training...")
        results = model.train(**train_args)
        return results
    except SyntaxError as e:
        # In case future versions reject an arg
        print("[error] SyntaxError while passing args:", e)
        raise
    except Exception as e:
        print("[error] Training failed:", e)
        traceback.print_exc()
        raise

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Path to preprocessed or seeded dataset root")
    p.add_argument("--weights", default="yolov11l.pt", help="Path to pretrained weights (.pt)")
    p.add_argument("--project_root", default="./yolo_training_runs", help="Project root for outputs")
    p.add_argument("--project_name", default="yolov11l_road_defects", help="Run name")
    p.add_argument("--mode", choices=("fresh", "resume", "fixonly"), default="fixonly", help="Mode of operation")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=12, help="Requested per-step batch (auto-batch may reduce)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--cache", choices=("none", "disk", "ram"), default="disk", help="Caching strategy")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max-batch-test", type=int, default=16, help="Upper bound for auto-batch probing")
    p.add_argument("--desired-eff-batch", type=int, default=16, help="Conceptual effective batch size target")
    p.add_argument("--no-autobatch", action="store_true", help="Disable auto-batch probing")
    p.add_argument("--no-grad-ckpt", action="store_true", help="Disable gradient checkpointing")
    p.add_argument("--resume-ultralytics", action="store_true", help="Pass resume=True to Ultralytics trainer")
    args = p.parse_args(argv)

    dataset = Path(args.dataset).resolve()
    if not dataset.exists():
        print(f"[error] dataset path not found: {dataset}")
        sys.exit(1)

    data_yaml = fix_or_create_data_yaml(dataset)

    missing = ensure_split_structure(dataset)
    if missing:
        print("[error] missing required folders/files:")
        for m in missing:
            print("  -", m)
        sys.exit(1)

    if args.mode == "fixonly":
        print("[ok] Data verified. Exiting.")
        return

    if args.mode == "resume":
        ckpt = locate_checkpoint(Path(args.project_root), args.project_name)
        if not ckpt:
            print(f"[error] No checkpoint found in {args.project_root}/{args.project_name}/weights.")
            sys.exit(1)
        weights_to_load = ckpt
        resume_ul = True
        print(f"[ok] Resuming from checkpoint: {weights_to_load}")
    else:
        wpath = Path(args.weights).resolve()
        if not wpath.exists():
            print(f"[error] weights file not found: {wpath}")
            sys.exit(1)
        weights_to_load = str(wpath)
        resume_ul = False
        print(f"[ok] Starting fresh from weights: {weights_to_load}")

    results = train_with_yolo(
        weights=weights_to_load,
        data_yaml=data_yaml,
        project_root=args.project_root,
        project_name=args.project_name,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        resume_ul=resume_ul,
        cache_option=args.cache,
        max_batch_test=args.max_batch_test,
        desired_eff_batch=args.desired_eff_batch,
        workers=args.workers,
        enable_autobatch=not args.no_autobatch,
        enable_checkpointing=not args.no_grad_ckpt,
        safe_low_batch=4
    )
    print("[done] Training finished.")
    return results

if __name__ == "__main__":
    main()