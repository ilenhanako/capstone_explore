#!/usr/bin/env python3
"""
Convert CVAT ANNOTATIONS (XML) into YOLO-style dataset for YOLOv11.

Usage:
    python convert_cvat_to_yolov11.py \
      --xml GROUNDTRUTH.xml \
      --images-dir /path/to/images \
      --out /path/to/output_dataset \
      --split 0.8 0.1 0.1 \
      --iou-thr 0.5 \
      --seed 42

This will produce:
out/
  images/train/*.jpg
  images/val/*.jpg
  images/test/*.jpg
  labels/train/*.txt
  labels/val/*.txt
  labels/test/*.txt
  data.yaml   # class names + paths to train/val
"""
import argparse
import xml.etree.ElementTree as ET
import os
import shutil
import random
from collections import defaultdict, Counter
from math import isclose

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", required=True, help="Path to CVAT/LabelStudio XML (GROUNDTRUTH.xml)")
    p.add_argument("--images-dir", required=True, help="Directory with the image files referenced in XML")
    p.add_argument("--out", required=True, help="Output dataset directory (YOLO format)")
    p.add_argument("--split", nargs=3, type=float, default=[0.8,0.1,0.1], help="Train/Val/Test ratios (sum to 1)")
    p.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for merging overlapping boxes")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--merge-strategy", choices=["majority","union"], default="majority",
                   help="How to merge overlapping annotations: majority = vote + tie-break; union = keep all boxes")
    return p.parse_args()

def iou(boxA, boxB):
    # boxes are (xtl, ytl, xbr, ybr)
    xa = max(boxA[0], boxB[0])
    ya = max(boxA[1], boxB[1])
    xb = min(boxA[2], boxB[2])
    yb = min(boxA[3], boxB[3])
    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter = inter_w * inter_h
    areaA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    areaB = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    union = areaA + areaB - inter
    if union <= 0:
        return 0.0
    return inter / union

def cluster_boxes(boxes, iou_thr=0.5):
    """
    Greedy clustering: take first box, gather all boxes with IoU >= thr into group, remove them, repeat.
    boxes: list of dicts with keys: label, xtl, ytl, xbr, ybr, source
    Returns list of clusters (lists of boxes)
    """
    clusters = []
    remaining = boxes.copy()
    while remaining:
        base = remaining.pop(0)
        cluster = [base]
        to_remove = []
        for i, other in enumerate(remaining):
            if iou((base['xtl'], base['ytl'], base['xbr'], base['ybr']),
                   (other['xtl'], other['ytl'], other['xbr'], other['ybr'])) >= iou_thr:
                cluster.append(other)
                to_remove.append(i)
        # remove by indices in reverse
        for idx in sorted(to_remove, reverse=True):
            remaining.pop(idx)
        clusters.append(cluster)
    return clusters

def choose_box_for_cluster(cluster):
    """
    Choose a single representative box and label for a cluster.
    Strategy:
     - majority label across cluster
     - if tie: prefer box with source == 'manual' over 'file' or other
     - else prefer largest area
    Returns a dict with label and coords.
    """
    labels = [b['label'] for b in cluster]
    cnt = Counter(labels)
    label, count = cnt.most_common(1)[0]
    # if tie between multiple labels? check
    tied = [lab for lab, c in cnt.items() if c == count]
    if len(tied) > 1:
        # prefer any box whose source == manual and label in tied
        manual_candidates = [b for b in cluster if b.get('source','') == 'manual' and b['label'] in tied]
        if manual_candidates:
            # pick largest manual candidate
            chosen = max(manual_candidates, key=lambda b: (b['xbr']-b['xtl'])*(b['ybr']-b['ytl']))
            return chosen
        # else choose largest box among tied labels
        candidates = [b for b in cluster if b['label'] in tied]
        chosen = max(candidates, key=lambda b: (b['xbr']-b['xtl'])*(b['ybr']-b['ytl']))
        return chosen
    else:
        # single majority label -> pick box with that label and prefer 'manual' then largest
        candidates = [b for b in cluster if b['label'] == label]
        manual = [b for b in candidates if b.get('source','') == 'manual']
        if manual:
            chosen = max(manual, key=lambda b: (b['xbr']-b['xtl'])*(b['ybr']-b['ytl']))
            return chosen
        chosen = max(candidates, key=lambda b: (b['xbr']-b['xtl'])*(b['ybr']-b['ytl']))
        return chosen

def parse_cvat_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    # classes
    label_list = []
    job = root.find('meta').find('job')
    for lab in job.find('labels').findall('label'):
        name = lab.find('name').text
        label_list.append(name)
    # gather image data
    images = []
    for img in root.findall('image'):
        img_id = img.get('id')
        name = img.get('name')
        width = float(img.get('width'))
        height = float(img.get('height'))
        boxes = []
        for box in img.findall('box'):
            label = box.get('label')
            source = box.get('source') or ''
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            boxes.append({'label':label,'source':source,'xtl':xtl,'ytl':ytl,'xbr':xbr,'ybr':ybr})
        images.append({'id':img_id,'name':name,'width':width,'height':height,'boxes':boxes})
    return label_list, images

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def write_yolo_label(txt_path, boxes, class_map, img_w, img_h):
    """
    boxes: list of dicts chosen representative boxes
    class_map: name->idx
    Writes YOLO txt file (one line per box): class x_center y_center width height (normalized)
    """
    lines = []
    for b in boxes:
        cls = class_map[b['label']]
        x_center = (b['xtl'] + b['xbr']) / 2.0 / img_w
        y_center = (b['ytl'] + b['ybr']) / 2.0 / img_h
        bw = (b['xbr'] - b['xtl']) / img_w
        bh = (b['ybr'] - b['ytl']) / img_h
        # clamp tiny values to zero
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))
        lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

def main():
    args = parse_args()
    random.seed(args.seed)
    if not isclose(sum(args.split), 1.0, rel_tol=1e-6):
        raise SystemExit("split ratios must sum to 1.0")

    classes, images = parse_cvat_xml(args.xml)
    class_map = {name:i for i,name in enumerate(classes)}
    out = args.out
    images_out = os.path.join(out, "images")
    labels_out = os.path.join(out, "labels")
    for subset in ("train","val","test"):
        ensure_dir(os.path.join(images_out, subset))
        ensure_dir(os.path.join(labels_out, subset))

    # split images by filenames deterministically
    image_names = [img['name'] for img in images]
    random.shuffle(image_names)
    n = len(image_names)
    n_train = int(n * args.split[0])
    n_val = int(n * args.split[1])
    train_set = set(image_names[:n_train])
    val_set = set(image_names[n_train:n_train+n_val])
    test_set = set(image_names[n_train+n_val:])

    # map image name -> image dict for quick lookup
    img_map = {img['name']: img for img in images}

    # iterate and produce labels
    for name in image_names:
        img = img_map[name]
        subset = "train" if name in train_set else ("val" if name in val_set else "test")
        src_path = os.path.join(args.images_dir, name)
        if not os.path.exists(src_path):
            print(f"Warning: image {name} not found in images-dir; skipping.")
            continue
        dst_img_path = os.path.join(images_out, subset, name)
        shutil.copy2(src_path, dst_img_path)

        boxes = img['boxes']
        chosen_boxes = []
        if args.merge_strategy == "union":
            # keep all (no merge) but remove zero-area boxes
            chosen_boxes = [b for b in boxes if (b['xbr']>b['xtl'] and b['ybr']>b['ytl'])]
        else:
            # cluster and choose one per cluster
            clusters = cluster_boxes(boxes, iou_thr=args.iou_thr)
            for c in clusters:
                if not c:
                    continue
                chosen = choose_box_for_cluster(c)
                # sanity check positive area
                if chosen['xbr']>chosen['xtl'] and chosen['ybr']>chosen['ytl']:
                    chosen_boxes.append(chosen)
        label_txt = os.path.splitext(name)[0] + ".txt"
        label_path = os.path.join(labels_out, subset, label_txt)
        write_yolo_label(label_path, chosen_boxes, class_map, img['width'], img['height'])

    # write data.yaml
    data_yaml = os.path.join(out, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"names: {classes}\n")
        f.write(f"nc: {len(classes)}\n")
        f.write("train: " + os.path.join(images_out, "train") + "\n")
        f.write("val: " + os.path.join(images_out, "val") + "\n")
        f.write("test: " + os.path.join(images_out, "test") + "\n")
    print("Done. Dataset written to", out)
    print("classes:", classes)
    print("data.yaml:", data_yaml)

if __name__ == "__main__":
    main()