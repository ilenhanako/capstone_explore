#!/usr/bin/env python3
"""
eval_from_images_recommended.py

Robust inference + COCO evaluation for Ultralytics YOLO models.

Features / behavior
- Loads an Ultralytics YOLO model and runs inference on images listed in a GT COCO JSON.
- Maps model class index -> model class name -> GT category id using a robust, normalized name mapping.
  - Supports a user-provided name-map (synonyms) JSON and an ignore-list for classes to drop.
  - Normalizes (strip + lower) names for matching.
- Optional GT id offset (0 or 1) if you need to convert GT category ids to COCO 1-based convention.
- Validates prediction bbox format (xywh, pixels), image_id presence, and bbox extents.
- Writes:
  - predictions_coco.json (COCO results)
  - mapping_report.json (what was mapped/skipped)
  - eval_summary.json (COCO mAP & per-class AP)
  - timing.json (inference timing stats)
- Prints human-friendly logs and mapping report.

Usage:
  pip install ultralytics pycocotools opencv-python tqdm numpy
  python eval_from_images_recommended.py \
      --model /path/to/best.pt \
      --gt-json gt_coco.json \
      --images-root /path/to/images \
      --out-dir results_eval \
      --device 0 \
      --conf 0.25 \
      --iou 0.45 \
      --name-map name_map.json \
      --ignore-classes road_rutting \
      --gt-id-offset 0

Notes:
- This script maps by name so two models with different numeric class ids but the same names are compared correctly.
- If a model class name is not found in GT (after normalization and synonyms), its predictions are skipped and reported.
"""

import argparse
import json
import os
import time
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except Exception as e:
    raise RuntimeError("pycocotools is required: pip install pycocotools") from e


def norm_name(s):
    return s.strip().lower() if isinstance(s, str) else s


def load_name_map(path):
    if not path:
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    # Normalize keys and values to lower-case stripped form
    out = {}
    for k, v in data.items():
        if k is None or v is None:
            continue
        out[k.strip()] = v.strip()
        out[norm_name(k)] = norm_name(v)
    return out


def load_model_ultralytics(model_path, device):
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("ultralytics package required: pip install ultralytics") from e

    model = YOLO(model_path)
    # attempt to move model to device if supported
    dev = "cpu"
    try:
        if device is not None and device != "cpu":
            # device may be int or "cuda:0" etc.
            if isinstance(device, int):
                devstr = f"cuda:{device}"
            else:
                devstr = str(device)
            model.to(devstr)
            dev = devstr
        else:
            dev = "cpu"
    except Exception:
        dev = "cpu"

    # Normalize model.names to dict idx->name
    names = {}
    if hasattr(model, "names"):
        try:
            names_raw = model.names
            if isinstance(names_raw, dict):
                names = {int(k): v for k, v in names_raw.items()}
            elif isinstance(names_raw, (list, tuple)):
                names = {i: v for i, v in enumerate(names_raw)}
        except Exception:
            names = {}
    return model, dev, names


def run_inference_and_build_raw_preds(model, images, images_root, conf=0.25, iou=0.45):
    """
    Runs inference for each image dict in images (contains 'id' and 'file_name').
    Returns raw predictions list with temporary category_id = model index, plus timing list.
    """
    raw_preds = []
    times = []
    for img in tqdm(images, desc="Inference"):
        image_id = int(img["id"])
        fname = img["file_name"]
        path = os.path.join(images_root, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

        t0 = time.perf_counter()
        try:
            res = model(path, conf=conf, iou=iou)
        except Exception as e:
            raise RuntimeError(f"Model inference failed on {path}: {e}")
        t1 = time.perf_counter()
        times.append(t1 - t0)

        # results may be a list-like; take first element
        r = res[0] if isinstance(res, (list, tuple)) and len(res) > 0 else res
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue

        # Extract arrays with robust fallbacks
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
        except Exception:
            # fallback: convert to numpy array if possible
            try:
                arr = np.array(boxes)
                if arr.size == 0:
                    continue
                xyxy = arr[:, :4]
                confs = arr[:, 4]
                cls = arr[:, 5].astype(int)
            except Exception:
                continue

        for (x1, y1, x2, y2), score, c in zip(xyxy, confs, cls):
            x = float(x1)
            y = float(y1)
            w = float(x2 - x1)
            h = float(y2 - y1)
            raw_preds.append({
                "image_id": image_id,
                "category_id": int(c),   # model class idx (temporary)
                "bbox_xywh": [x, y, w, h],
                "score": float(score)
            })

    return raw_preds, times


def build_model_idx_to_canonical(model_names, name_map):
    """
    Build mapping model_idx -> canonical normalized name.
    Priority:
      - exact model_name as key in name_map -> mapped value
      - normalized model_name in name_map -> mapped value
      - otherwise normalized model_name itself
    """
    out = {}
    for idx, raw in model_names.items():
        if raw is None:
            continue
        # try exact key on original raw
        canonical = None
        if raw in name_map:
            canonical = name_map[raw]
        elif norm_name(raw) in name_map:
            canonical = name_map[norm_name(raw)]
        else:
            canonical = raw
        out[int(idx)] = norm_name(canonical)
    return out


def map_preds_to_gt_ids(raw_preds, model_idx_to_canonical, gt_name_to_id, ignore_set=None, gt_id_offset=0):
    """
    Map raw preds (with model idx) -> COCO-style preds using GT category ids.
    Returns mapped_preds (list), and a mapping_report dict.
    """
    ignore_set = {norm_name(s) for s in (ignore_set or [])}

    # Build normalized GT map (norm_name -> id)
    gt_norm_to_id = {norm_name(k): v for k, v in gt_name_to_id.items()}

    mapped = []
    report = {
        "total_raw_preds": len(raw_preds),
        "skipped_unknown_model_idx": 0,
        "skipped_ignored": 0,
        "skipped_no_gt_match": 0,
        "per_model_idx_counts": Counter(),
        "per_canonical_no_gt": Counter()
    }

    for p in raw_preds:
        midx = int(p["category_id"])
        canonical = model_idx_to_canonical.get(midx, None)
        report["per_model_idx_counts"][str(midx)] += 1
        if canonical is None:
            report["skipped_unknown_model_idx"] += 1
            continue
        if canonical in ignore_set:
            report["skipped_ignored"] += 1
            continue
        gt_id = gt_norm_to_id.get(canonical, None)
        if gt_id is None:
            report["skipped_no_gt_match"] += 1
            report["per_canonical_no_gt"][canonical] += 1
            continue
        mapped.append({
            "image_id": int(p["image_id"]),
            "category_id": int(gt_id) + int(gt_id_offset),
            "bbox": [float(x) for x in p["bbox_xywh"]],
            "score": float(p["score"])
        })

    # convert counters to plain dicts
    report["per_model_idx_counts"] = dict(report["per_model_idx_counts"])
    report["per_canonical_no_gt"] = dict(report["per_canonical_no_gt"])
    return mapped, report


def validate_predictions_against_gt(preds, gt, images_root, throw_on_error=False):
    """
    Basic validation checks:
      - Each pred.image_id exists in GT images
      - bbox fields are 4 numbers with w>0,h>0
      - bbox lies (partially) within image bounds (warn otherwise)
    Returns list of warnings (strings).
    """
    warnings = []
    img_map = {int(i["id"]): i for i in gt.get("images", [])}
    for p in preds:
        iid = int(p["image_id"])
        if iid not in img_map:
            msg = f"prediction references unknown image_id {iid}"
            warnings.append(msg)
            if throw_on_error:
                raise ValueError(msg)
            continue
        img = img_map[iid]
        W, H = int(img.get("width", 0)), int(img.get("height", 0))
        bx = p.get("bbox", None)
        if not bx or len(bx) != 4:
            msg = f"invalid bbox for pred image_id {iid}: {bx}"
            warnings.append(msg)
            if throw_on_error:
                raise ValueError(msg)
            continue
        x, y, w, h = [float(v) for v in bx]
        if w <= 0 or h <= 0:
            warnings.append(f"bbox non-positive w/h for image_id {iid}: {bx}")
        # check overlap
        if (x + w) < 0 or (y + h) < 0 or x > W or y > H:
            warnings.append(f"bbox outside image bounds for image_id {iid}: bbox={bx} img_wh=({W},{H})")
    return warnings


def evaluate_coco(gt_json, preds_json, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(preds_json)

    ce = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType="bbox")
    ce.evaluate()
    ce.accumulate()
    ce.summarize()

    stats = getattr(ce, "stats", None)
    summary = {}
    if stats is None or len(stats) < 8:
        summary["AP_50_95"] = None
        summary["AP_50"] = None
    else:
        summary["AP_50_95"] = float(stats[0])
        summary["AP_50"] = float(stats[1])
        summary["AP_75"] = float(stats[2])

    # per-class
    per_class = {}
    cat_ids = coco_gt.getCatIds()
    cats = coco_gt.loadCats(cat_ids)
    name_by_id = {c["id"]: c["name"] for c in cats}
    for cid in cat_ids:
        ann_ids = coco_gt.getAnnIds(catIds=[cid])
        n_gt = len(ann_ids)
        try:
            pred_ann_ids = coco_dt.getAnnIds(catIds=[cid])
            n_preds = len(pred_ann_ids)
        except Exception:
            n_preds = 0
        # evaluate per-class
        ce_cls = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType="bbox")
        ce_cls.params.useCats = 1
        ce_cls.params.catIds = [cid]
        ce_cls.evaluate()
        ce_cls.accumulate()
        stats_cls = getattr(ce_cls, "stats", None)
        ap = stats_cls[0] if stats_cls is not None and len(stats_cls) > 0 else None
        ap50 = stats_cls[1] if stats_cls is not None and len(stats_cls) > 1 else None
        per_class[name_by_id[cid]] = {"AP": ap, "AP50": ap50, "n_gt": n_gt, "n_preds": n_preds}

    out = {
        "summary": summary,
        "per_class": per_class
    }
    out_path = os.path.join(out_dir, "eval_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def summarise_times(times):
    arr = np.array(times)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "p95_s": float(np.percentile(arr, 95)),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
        "fps": float(arr.size / arr.sum()) if arr.sum() > 0 else None
    }


def main():
    p = argparse.ArgumentParser(description="Robust inference + COCO evaluation")
    p.add_argument("--model", required=True, help="Path to model .pt")
    p.add_argument("--gt-json", required=True, help="GT COCO json")
    p.add_argument("--images-root", required=True, help="Root folder with images")
    p.add_argument("--out-dir", default="results_eval", help="Output directory")
    p.add_argument("--device", default=0, help="Device id (0) or 'cpu' or 'cuda:0'")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--name-map", required=False, help="JSON file mapping model names -> GT canonical names")
    p.add_argument("--ignore-classes", required=False, help="Comma-separated list of model class names to ignore (pre-normalized)")
    p.add_argument("--gt-id-offset", type=int, choices=[0, 1], default=0, help="Offset to add to GT ids when writing preds (0 or 1)")
    args = p.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # load GT
    with open(args.gt_json, "r") as f:
        gt = json.load(f)
    gt_images = gt.get("images", [])
    gt_cats = gt.get("categories", [])
    gt_name_to_id = {c["name"]: int(c["id"]) for c in gt_cats}

    print("GT categories (id->name):")
    for c in gt_cats:
        print(f"  {c['id']} {c['name']}")

    # load name map
    name_map = load_name_map(args.name_map) if args.name_map else {}
    if name_map:
        print("Loaded name_map entries:", len(name_map))

    ignore_list = []
    if args.ignore_classes:
        ignore_list = [s.strip() for s in args.ignore_classes.split(",") if s.strip()]
        print("Ignore classes:", ignore_list)

    # load model
    print("Loading model...")
    model, device, model_names = load_model_ultralytics(args.model, args.device)
    print(f"Model loaded on device: {device}. Model classes: {len(model_names)}")
    print("Model class names (index -> name):")
    for i, n in model_names.items():
        print(f"  {i}: {n}")

    # build canonical mapping from model indices -> canonical normalized name
    model_idx_to_canonical = build_model_idx_to_canonical(model_names, name_map)

    # run inference
    raw_preds, times = run_inference_and_build_raw_preds(model, gt_images, args.images_root, conf=args.conf, iou=args.iou)
    print(f"Inference done. Raw predictions: {len(raw_preds)}")

    # map to GT ids
    mapped_preds, mapping_report = map_preds_to_gt_ids(raw_preds, model_idx_to_canonical, gt_name_to_id, ignore_set=ignore_list, gt_id_offset=args.gt_id_offset)
    print(f"Mapped predictions: {len(mapped_preds)}")
    print("Mapping report:", json.dumps(mapping_report, indent=2))

    # save mapping report
    with open(os.path.join(out_dir, "mapping_report.json"), "w") as f:
        json.dump(mapping_report, f, indent=2)

    # validate predictions
    validation_warnings = validate_predictions_against_gt(mapped_preds, gt, args.images_root)
    if validation_warnings:
        print("Validation warnings (sample up to 20):")
        for w in validation_warnings[:20]:
            print("  -", w)
    with open(os.path.join(out_dir, "validation_warnings.json"), "w") as f:
        json.dump(validation_warnings, f, indent=2)

    # write predictions file (COCO results format)
    preds_path = os.path.join(out_dir, "predictions_coco.json")
    with open(preds_path, "w") as f:
        json.dump(mapped_preds, f)
    print("Saved predictions to:", preds_path)

    # run COCO evaluation
    print("Running COCO evaluation...")
    summary = evaluate_coco(args.gt_json, preds_path, out_dir)
    print("Saved eval summary to eval_summary.json")

    # timing
    time_stats = summarise_times(times)
    with open(os.path.join(out_dir, "timing.json"), "w") as f:
        json.dump(time_stats, f, indent=2)
    print("Timing summary:", time_stats)

    print("Done. Outputs in:", out_dir)


if __name__ == "__main__":
    main()