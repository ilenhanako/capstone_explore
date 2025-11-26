#!/usr/bin/env python3
"""
Two-phase transfer learning / fine-tuning script for YOLOv11 using Ultralytics API.

Updates:
- TRANSFER_HPARAMS adjusted for bbox-only datasets: copy_paste disabled and small rotation ('degrees') added.
- Phase 2 LR reduction: Phase 2 lr0 is reduced by 10x before fine-tuning to protect pretrained backbone weights.

Notes:
- copy_paste in TRANSFER_HPARAMS is set to 0.0 because copy-paste augmentation generally requires instance masks/polygons.
  If you *do* have segmentation masks and want to enable copy_paste, set this to a positive value and ensure your train() accepts it.
- The script still uses best-effort application of TRANSFER_HPARAMS to model.model.hyp and best-effort config of loss.
"""
import argparse
import os
import glob
import shutil
import time

try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit("Please install ultralytics package (pip install ultralytics). Exception: " + str(e))

import torch
import torch.nn as nn

# Transfer-specific hyperparameters suggested by the user (adjusted)
TRANSFER_HPARAMS = {
    # loss gains (keep same as OPTIMIZED_HPARAMS in your workflow)
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,

    # TRANSFER LEARNING SPECIFIC:
    'lr0': 0.001,       # Phase 1 head LR (will be reduced for Phase 2)
    'lrf': 0.1,         # Final LR will be lr0 * lrf
    'warmup_epochs': 3, # Shorter warmup (was 5)
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,

    # Aggressive Augmentation for Small Dataset (1200 images)
    'mosaic': 1.0,      # Keep high (1.0 => on)
    'mixup': 0.2,       # Slight increase to force generalization (0.0..1.0)
    'copy_paste': 0.0,  # Disabled by default for bbox-only datasets (requires masks/polygons)
    'degrees': 10.0,    # small rotation augmentation to compensate
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to data.yaml")
    p.add_argument("--weights", default="yolov11m.pt", help="Pretrained weights")
    p.add_argument("--phase1-epochs", type=int, default=5, help="Head-only warmup epochs (freeze backbone)")
    p.add_argument("--phase2-epochs", type=int, default=40, help="Full fine-tune epochs after unfreeze")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--project", default="transfer_learning", help="project dir for ultralytics output")
    p.add_argument("--name", default="exp", help="run name")
    p.add_argument("--device", default=None, help="device (e.g., 0, 'cpu') or leave None for auto")
    p.add_argument("--freeze-bn", action="store_true", help="Put frozen backbone BNs into eval() mode (freeze running stats)")
    p.add_argument("--ema", action="store_true", help="Enable EMA averaging if supported by the training wrapper")
    p.add_argument("--focal-gamma", type=float, default=0.0, help="If >0, attempt to set focal loss gamma (best-effort)")
    p.add_argument("--no-apply-hyp", action="store_true", help="Do not apply TRANSFER_HPARAMS to model.hyp (for debugging)")
    return p.parse_args()

def freeze_backbone_best_effort(model):
    module = getattr(model, "model", None)
    if module is None:
        print("Warning: could not find 'model' attribute on YOLO instance — skipping freeze.")
        return

    # Try to freeze common backbone attributes
    frozen_params = 0
    backbone_candidates = []
    for name in ["backbone", "backbone_layers", "model", "features"]:
        if hasattr(module, name) and name != "model":
            backbone_candidates.append(name)

    if backbone_candidates:
        for attr in backbone_candidates:
            sub = getattr(module, attr)
            for p in sub.parameters():
                p.requires_grad = False
                frozen_params += 1
        print(f"Froze parameters in backbone attributes: {backbone_candidates} (approx frozen params: {frozen_params})")
        return

    # fallback: freeze ~60% of parameters
    params = list(module.named_parameters())
    n = len(params)
    freeze_n = int(0.6 * n)
    for i, (name, p) in enumerate(params):
        if i < freeze_n:
            p.requires_grad = False
        else:
            p.requires_grad = True
    print(f"Backbone freeze heuristic: froze first {freeze_n}/{n} param groups (approx 60%)")
    return

def set_frozen_bn_eval(model):
    module = getattr(model, "model", None)
    if module is None:
        return
    def _set_eval(m):
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            m.eval()
    module.apply(_set_eval)
    print("Set BatchNorm layers in frozen backbone to eval() (best-effort).")

def unfreeze_all(model):
    module = getattr(model, "model", None)
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = True
    print("Unfroze all model parameters.")

def find_best_checkpoint(project, name):
    base = os.path.join(project, name, "weights")
    candidates = []
    if os.path.isdir(base):
        candidates = glob.glob(os.path.join(base, "*.pt"))
    if not candidates:
        exp_dir = os.path.join(project, name)
        if os.path.isdir(exp_dir):
            candidates = glob.glob(os.path.join(exp_dir, "weights", "*.pt"))
    if not candidates:
        return None
    for c in candidates:
        if os.path.basename(c) == "best.pt":
            return c
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]

def try_configure_loss(model, focal_gamma=0.0, hyp=None):
    module = getattr(model, "model", None)
    if module is None:
        print("Warning: no internal model found to configure loss/hyp.")
        return

    # Apply hyp (loss gains, augmentation strengths) if model exposes 'hyp' dict
    if hyp and hasattr(module, "hyp"):
        try:
            old_hyp = getattr(module, "hyp")
            # update only keys present in hyp
            for k, v in hyp.items():
                old_hyp[k] = v
            module.hyp = old_hyp
            print("Applied TRANSFER_HPARAMS to model.model.hyp (best-effort).")
        except Exception as e:
            print("Could not apply hyp to model.model.hyp:", e)
    else:
        if hyp:
            print("model.model.hyp not found; hyp not applied to internal model.")

    # Try to set focal gamma on internal loss object
    loss_obj = getattr(module, "loss", None)
    if loss_obj is None:
        if focal_gamma > 0.0:
            print("No accessible loss object to set focal gamma on (skipping).")
        return
    if focal_gamma and hasattr(loss_obj, "fl_gamma"):
        try:
            old = getattr(loss_obj, "fl_gamma")
            setattr(loss_obj, "fl_gamma", float(focal_gamma))
            print(f"Configured loss.fl_gamma: {old} -> {focal_gamma}")
        except Exception as e:
            print("Couldn't set loss.fl_gamma:", e)
    else:
        if focal_gamma:
            print("Loss object does not expose 'fl_gamma'; focal parameter not applied.")

def build_train_kwargs(trans_hyp, args, phase_name):
    """
    Build kwargs for model.train() combining CLI args and trans_hyp.
    We pass as many keys as we can; ultralytics.train will ignore unknown kwargs,
    but older/newer versions might error — this is best-effort.
    """
    train_kwargs = dict(
        data=args.data,
        epochs=args.phase1_epochs if "phase1" in phase_name else args.phase2_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=phase_name,
        device=args.device,
    )

    # LR schedule keys from hyp
    if "lr0" in trans_hyp:
        train_kwargs["lr0"] = trans_hyp["lr0"]
    if "lrf" in trans_hyp:
        train_kwargs["lrf"] = trans_hyp["lrf"]
    if "warmup_epochs" in trans_hyp:
        train_kwargs["warmup_epochs"] = trans_hyp["warmup_epochs"]
    if "warmup_momentum" in trans_hyp:
        train_kwargs["warmup_momentum"] = trans_hyp["warmup_momentum"]
    if "warmup_bias_lr" in trans_hyp:
        train_kwargs["warmup_bias_lr"] = trans_hyp["warmup_bias_lr"]

    # augmentation/flags - some accept float strengths, others boolean.
    if "mosaic" in trans_hyp:
        train_kwargs["mosaic"] = bool(trans_hyp["mosaic"])
    if "mixup" in trans_hyp:
        train_kwargs["mixup"] = float(trans_hyp["mixup"]) if isinstance(trans_hyp["mixup"], (int,float)) else bool(trans_hyp["mixup"])
    if "copy_paste" in trans_hyp:
        train_kwargs["copy_paste"] = float(trans_hyp["copy_paste"])
    if "degrees" in trans_hyp:
        train_kwargs["degrees"] = float(trans_hyp["degrees"])

    # EMA if requested by CLI
    if args.ema:
        train_kwargs["ema"] = True

    return train_kwargs

def run_phase(model, trans_hyp, args, phase_epochs, phase_name):
    train_kwargs = build_train_kwargs(trans_hyp, args, phase_name)
    train_kwargs["epochs"] = phase_epochs

    print(f"Running {phase_name} with train kwargs:")
    for k, v in sorted(train_kwargs.items()):
        print(f"  {k}: {v}")

    model.train(**train_kwargs)

def main():
    args = parse_args()

    # Ensure unique run names for phases
    phase1_name = args.name + "_phase1_head"
    phase2_name = args.name + "_phase2_finetune"

    print("Loading model from weights:", args.weights)
    model = YOLO(args.weights)

    # Apply TRANSFER_HPARAMS to internal model.hyp/loss where possible (best-effort)
    if not args.no_apply_hyp:
        try_configure_loss(model, focal_gamma=args.focal_gamma, hyp=TRANSFER_HPARAMS)
    else:
        print("Skipping applying TRANSFER_HPARAMS to model internals (--no-apply-hyp)")

    # Phase 1: freeze backbone and train head-only
    print("Phase 1: freezing backbone (best-effort) and training head-only for", args.phase1_epochs, "epochs")
    freeze_backbone_best_effort(model)
    if args.freeze_bn:
        set_frozen_bn_eval(model)

    phase1_hyp = dict(TRANSFER_HPARAMS)
    phase1_epochs = args.phase1_epochs
    run_phase(model, phase1_hyp, args, phase1_epochs, phase1_name)

    # locate checkpoint for phase1
    print("Locating best checkpoint from Phase 1 run...")
    ckpt1 = find_best_checkpoint(args.project, phase1_name)
    if ckpt1 is None:
        print("Warning: no checkpoint found for phase1; attempting to use last or provided weights.")
        ckpt1 = find_best_checkpoint(args.project, phase1_name) or args.weights
    print("Phase 1 checkpoint:", ckpt1)

    # Phase 2: reload checkpoint, unfreeze, apply hyp, and finetune
    print("Phase 2: loading checkpoint and unfreezing all parameters for fine-tuning")
    model = YOLO(ckpt1)
    unfreeze_all(model)

    # Re-apply hyp after reload
    if not args.no_apply_hyp:
        try_configure_loss(model, focal_gamma=args.focal_gamma, hyp=TRANSFER_HPARAMS)

    phase2_hyp = dict(TRANSFER_HPARAMS)

    # --- Critical Adjustment: Lower Phase 2 Learning Rate to protect pretrained backbone ---
    if "lr0" in phase2_hyp:
        old_lr = phase2_hyp["lr0"]
        phase2_hyp["lr0"] = float(old_lr) / 10.0
        print(f"Reducing LR for Phase 2 from {old_lr} to {phase2_hyp['lr0']} to avoid destroying backbone features")

    phase2_epochs = args.phase2_epochs
    run_phase(model, phase2_hyp, args, phase2_epochs, phase2_name)

    print("Training complete. Check runs directory for results.")
    print("Phase1 run name:", phase1_name)
    print("Phase2 run name:", phase2_name)
    print("Final TRANSFER_HPARAMS applied (best-effort):")
    for k, v in sorted(TRANSFER_HPARAMS.items()):
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()