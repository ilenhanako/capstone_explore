import argparse
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import sys
import threading
import queue
import time
import logging
from boxmot import ByteTrack
try:
    from boxmot import ByteTrack
    ByteTrack_AVAILABLE = True
except Exception:
    ByteTrack_AVAILABLE = False
from ultralytics import YOLO
import torch
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("road_defect_pipeline")

CLASS_NAMES = {
    0: "road_crack_longitudinal",
    1: "road_crack_transverse",
    2: "road_crack_alligator",
    3: "pothole",
    4: "marking_faded",
    5: "distractor_manhole",
    6: "distractor_patch"
}

COLORS = {
    0: (0, 255, 0),
    1: (255, 0, 0),
    2: (0, 0, 139),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (128, 0, 128),
    6: (255, 128, 0),
}

###FALLBACK TRACK JIC
class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_object_id = 0
        self.objects = dict()
        self.bboxes = dict()
        self.disappeared = dict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox):
        oid = self.next_object_id
        self.objects[oid] = centroid
        self.bboxes[oid] = bbox
        self.disappeared[oid] = 0
        self.next_object_id += 1
        return oid

    def deregister(self, oid):
        self.objects.pop(oid, None)
        self.bboxes.pop(oid, None)
        self.disappeared.pop(oid, None)

    def update(self, detections):
        if len(detections) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return [[*self.bboxes[oid], oid, None, None] for oid in self.bboxes.keys()]
        input_centroids = []
        input_bboxes = []
        input_scores = []
        input_cls = []
        for det in detections:
            x1,y1,x2,y2,score,cls = det
            cx = int((x1+x2)/2); cy = int((y1+y2)/2)
            input_centroids.append((cx,cy)); input_bboxes.append((int(x1),int(y1),int(x2),int(y2)))
            input_scores.append(score); input_cls.append(cls)
        if len(self.objects) == 0:
            out=[]
            for i,c in enumerate(input_centroids):
                oid = self.register(c, input_bboxes[i])
                out.append([*input_bboxes[i], oid, input_cls[i], input_scores[i]])
            return out
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid] for oid in object_ids]
        D = np.zeros((len(object_centroids), len(input_centroids)), dtype=np.float32)
        for i,oc in enumerate(object_centroids):
            for j,ic in enumerate(input_centroids):
                D[i,j] = np.hypot(oc[0]-ic[0], oc[1]-ic[1])
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        used_rows=set(); used_cols=set(); out=[]
        for row,col in zip(rows,cols):
            if row in used_rows or col in used_cols: continue
            if D[row,col] > self.max_distance: continue
            oid = object_ids[row]
            self.objects[oid] = input_centroids[col]
            self.bboxes[oid] = input_bboxes[col]
            self.disappeared[oid]=0
            used_rows.add(row); used_cols.add(col)
            out.append([*input_bboxes[col], oid, input_cls[col], input_scores[col]])
        unused_rows = set(range(0,D.shape[0])) - used_rows
        for row in unused_rows:
            oid = object_ids[row]
            self.disappeared[oid]+=1
            if self.disappeared[oid] > self.max_disappeared: self.deregister(oid)
        unused_cols = set(range(0,D.shape[1])) - used_cols
        for col in unused_cols:
            oid = self.register(input_centroids[col], input_bboxes[col])
            out.append([*input_bboxes[col], oid, input_cls[col], input_scores[col]])
        return out



#FRAME EXTRACTOR, quality gating based on blur and exposure
class FrameExtractor:
    def __init__(self, target_fps: int = None, blur_threshold: float = 80.0,
                 brightness_min: float = 10.0, brightness_max: float = 250.0,
                 motion_threshold: float = 0.025):
        self.target_fps = target_fps
        self.blur_threshold = blur_threshold
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.motion_threshold = motion_threshold
        self.stats = defaultdict(int)

    def is_blurry(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < self.blur_threshold, laplacian_var

    def is_overexposed(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(gray.mean())
        is_bad = (mean_brightness < self.brightness_min or mean_brightness > self.brightness_max)
        return is_bad, mean_brightness

    def compute_motion(self, frame, prev_frame):
        if prev_frame is None:
            return 1.0
        g1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(g1, g2)
        return float(diff.mean() / 255.0)

    def should_accept_frame(self, frame, prev_frame=None):
        metrics = {}
        is_blur, blur_var = self.is_blurry(frame)
        metrics['blur_variance'] = blur_var
        if is_blur:
            return False, 'blur', metrics
        is_bad_exposure, brightness = self.is_overexposed(frame)
        metrics['brightness'] = brightness
        if is_bad_exposure:
            return False, 'brightness', metrics
        if prev_frame is not None:
            motion = self.compute_motion(frame, prev_frame)
            metrics['motion'] = motion
            if motion < self.motion_threshold:
                return False, 'motion', metrics
        return True, 'accepted', metrics

    def extract_frames(self, video_path, verbose=True):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if self.target_fps and self.target_fps < source_fps:
            frame_interval = max(1, int(round(source_fps / self.target_fps)))
        else:
            frame_interval = 1

        if verbose:
            logger.info(f"Video: {Path(video_path).name} | source_fps={source_fps:.2f} | interval={frame_interval} | total_frames={total_frames}")

        frame_idx = 0
        prev_frame = None
        with tqdm(total=total_frames, desc="Extracting frames", disable=not verbose) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self.stats['total_frames'] += 1
                if frame_idx % frame_interval == 0:
                    self.stats['sampled_frames'] += 1
                    accept, reason, metrics = self.should_accept_frame(frame, prev_frame)
                    if accept:
                        self.stats['accepted_frames'] += 1
                        yield frame_idx, frame.copy(), metrics
                        prev_frame = frame.copy()
                    else:
                        self.stats[f'rejected_{reason}'] += 1
                frame_idx += 1
                pbar.update(1)
        cap.release()
        if verbose:
            logger.info(f"Frame extraction complete. sampled={self.stats['sampled_frames']} accepted={self.stats['accepted_frames']}")

# ---------------------------------------------------------------------------
#preprocessing using clahe, improving detail in low-contrast areas without excessively amplifying noise

##(TODO) LAINE TO COMBINE
# ---------------------------------------------------------------------------
class ImagePreprocessor:
    def __init__(self, target_size: int = 640, enable_clahe: bool = True,
                 clahe_clip_limit: float = 2.0, clahe_tile_size: int = 8):
        self.target_size = target_size
        self.enable_clahe = enable_clahe
        if enable_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_tile_size, clahe_tile_size))

    def resize_image(self, image):
        h, w = image.shape[:2]
        scale = self.target_size / max(h, w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = (self.target_size - new_w) // 2
        pad_h = (self.target_size - new_h) // 2
        top = pad_h
        bottom = self.target_size - new_h - pad_h
        left = pad_w
        right = self.target_size - new_w - pad_w
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
        return padded, scale, (left, top)

    def apply_clahe(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = self.clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        return enhanced

    def preprocess(self, image, return_original=False):
        original = image.copy() if return_original else None
        resized, scale, pad = self.resize_image(image)
        enhanced = self.apply_clahe(resized) if self.enable_clahe else resized
        metadata = {'scale': scale, 'pad': pad, 'original_shape': image.shape[:2], 'processed_shape': enhanced.shape[:2]}
        if return_original:
            return enhanced, metadata, original
        return enhanced, metadata

# ---------------------------------------------------------------------------
#visualisation
# ---------------------------------------------------------------------------
class Visualizer:
    @staticmethod
    def draw_tracks(frame, tracks, show_trajectory=False, trajectory_length=30):
        out = frame.copy()
        for tr in tracks:
            x1,y1,x2,y2,tid,cls,conf = tr
            col = COLORS.get(cls, (255,255,255))
            cv2.rectangle(out, (int(x1),int(y1)), (int(x2),int(y2)), col, 2)
            label = f"ID:{int(tid)} {CLASS_NAMES.get(int(cls),str(cls))} {float(conf):.2f}"
            (w,h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (int(x1), int(y1)-20), (int(x1)+w, int(y1)), col, -1)
            cv2.putText(out, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return out

    @staticmethod
    def create_comparison_view(original, enhanced, tracks):
        orig_annot = Visualizer.draw_tracks(original.copy(), tracks)
        enh_annot = Visualizer.draw_tracks(enhanced.copy(), tracks)
        cv2.putText(orig_annot, "Original", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(enh_annot, "Enhanced (CLAHE)", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        return np.hstack([orig_annot, enh_annot])



#change this to use other tracker. MIGHT WANT TO TRAIN REID for botsort since domain has high amts of occlusions (TODO)
class TrackerAdapter:
    def __init__(self, reid_weights, device):
        self.device = device
        if ByteTrack_AVAILABLE:
            try:
                self.tracker = ByteTrack( track_thresh=0.25,min_hits=1, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  
                self.backend = "bytetrack"
                logger.info("Using ByteTrack backend")
            except Exception as e:
                logger.warning(f"BoTSORT init failed: {e}. Falling back to centroid.")
                self.tracker = CentroidTracker()
                self.backend = "centroid"
        else:
            logger.info("BoTSORT not available; using centroid tracker")
            self.tracker = CentroidTracker()
            self.backend = "centroid"

    def update(self, detections, frame=None):
        if self.backend == "centroid":
            return self.tracker.update(list(detections))
        try:
            res = self.tracker.update(detections, frame)
        except TypeError:
            res = self.tracker.update(detections)
        except Exception as e:
            logger.warning(f"BoTSORT update error: {e}")
            return []
        out=[]
        try:
            for t in res:
                if isinstance(t, (list,tuple,np.ndarray)):
                    arr = list(t)
                    if len(arr) >= 6:
                        x1,y1,x2,y2 = float(arr[0]),float(arr[1]),float(arr[2]),float(arr[3])
                        tid = int(arr[4])
                        score = float(arr[5]) if len(arr)>5 else 0.0
                        cls = int(arr[6]) if len(arr)>6 else -1
                        out.append([x1,y1,x2,y2,tid,cls,score])
                        continue
                if isinstance(t, dict):
                    bbox = t.get('bbox') or t.get('tlbr') or t.get('box')
                    tid = t.get('track_id') or t.get('id')
                    score = t.get('score') or t.get('conf')
                    cls = t.get('cls') or t.get('class')
                    if bbox is not None and tid is not None:
                        x1,y1,x2,y2 = bbox[:4]
                        out.append([float(x1),float(y1),float(x2),float(y2),int(tid),int(cls) if cls is not None else -1,float(score) if score is not None else 0.0])
                        continue
                tid = getattr(t,'track_id',None) or getattr(t,'id',None)
                bbox = getattr(t,'tlbr',None) or getattr(t,'bbox',None)
                score = getattr(t,'score',None) or getattr(t,'conf',None)
                cls = getattr(t,'cls',None) or getattr(t,'class',None)
                if tid is not None and bbox is not None:
                    x1,y1,x2,y2 = bbox[:4]
                    out.append([float(x1),float(y1),float(x2),float(y2),int(tid),int(cls) if cls is not None else -1,float(score) if score is not None else 0.0])
            return out
        except Exception as e:
            logger.warning(f"TrackerAdapter normalization error: {e}")
            return []

class AsyncWriter(threading.Thread):
    """
    Async writer that writes:
      - Annotated frames to a video (if writer initialized)
      - Annotated images (single file per frame) into a flat directory when 'labeled' items are queued.
    Behavior:
      - For 'labeled' items we expect item = {'type':'labeled', 'frame_idx': int, 'tracks': list, 'frame': annotated_frame}
      - Only saves annotated_frame when tracks is non-empty.
    """
    def __init__(self, video_path=None, video_fps=30, frame_size=None, save_labeled_dir: Path=None, queue_size=512):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.video_fps = video_fps
        self.frame_size = frame_size
        self.save_labeled_dir = Path(save_labeled_dir) if save_labeled_dir is not None else None
        self.queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self.writer = None
        if self.video_path and self.frame_size:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(str(self.video_path), fourcc, self.video_fps, self.frame_size)
            if not self.writer.isOpened():
                logger.warning(f"VideoWriter couldn't open {self.video_path}; falling back to no-video output")
                self.writer = None
        # create flat labeled folder if requested
        if self.save_labeled_dir:
            self.save_labeled_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        logger.info("AsyncWriter started")
        while not self._stop_event.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                typ = item.get('type')
                if typ == 'frame' and self.writer is not None:
                    # Write annotated frame to video
                    frame = item['frame']
                    self.writer.write(frame)

                elif typ == 'labeled' and self.save_labeled_dir:
                    # Expect an annotated_frame already (with drawings), and a list of tracks
                    frame_idx = item.get('frame_idx')
                    tracks = item.get('tracks', [])
                    annotated = item.get('frame')  # already annotated_frame

                    # only save if there is at least one detection / track
                    if not tracks:
                        # skip saving empty frames
                        logger.debug(f"Skipping saving labeled image for frame {frame_idx} (no tracks).")
                        continue

                    # Save annotated image only (flat directory)
                    fname = f"annotated_frame_{int(frame_idx):06d}_tracks{len(tracks)}.jpg"
                    out_path = self.save_labeled_dir / fname
                    cv2.imwrite(str(out_path), annotated)

                    # NOTE: crops are intentionally not saved per your request.
                # ignore other types
            except Exception as e:
                logger.exception(f"AsyncWriter error: {e}")
            finally:
                try:
                    self.queue.task_done()
                except Exception:
                    pass

        # cleanup
        if self.writer is not None:
            self.writer.release()
        logger.info("AsyncWriter stopped")

    def stop(self):
        self._stop_event.set()

    def put(self, item, block=True, timeout=1.0):
        try:
            self.queue.put(item, block=block, timeout=timeout)
            return True
        except queue.Full:
            logger.warning("AsyncWriter queue full; dropping item")
            return False


def parse_ultralytics_result(res, conf_thresh=0.25):
    out = []
    if res is None:
        return out
    boxes = getattr(res, "boxes", None)
    if boxes is None:
        return out
    try:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy()
    except Exception:
        try:
            xyxy = []
            confs = []
            clss = []
            for b in boxes:
                xy = getattr(b, "xyxy", None)
                c = getattr(b, "conf", None)
                cl = getattr(b, "cls", None)
                if xy is None:
                    continue
                arr = xy[0].cpu().numpy() if hasattr(xy, 'cpu') else np.array(xy)
                xyxy.append(arr)
                confs.append(float(c[0]) if hasattr(c, '__len__') else float(c))
                clss.append(int(cl[0]) if hasattr(cl, '__len__') else int(cl))
            if len(xyxy) == 0:
                return out
            xyxy = np.array(xyxy)
            confs = np.array(confs)
            clss = np.array(clss)
        except Exception:
            return out
    for (x1,y1,x2,y2), conf, cls in zip(xyxy, confs, clss):
        if float(conf) < conf_thresh:
            continue
        out.append((float(x1), float(y1), float(x2), float(y2), float(conf), int(cls)))
    return out

# ---------------------------------------------------------------------------
#RoadDefectPipeline using batch inference and async writer(be aware of hardware req please))
# ---------------------------------------------------------------------------
class RoadDefectPipeline:
    def __init__(self, model_path: str, reid_weights: str = None, target_fps: int = None,
                 target_size: int = 640, enable_clahe: bool = True, conf_threshold: float = 0.5,
                 iou_threshold: float = 0.5, blur_threshold: float = 100.0, device: str = 'auto',
                 batch_size: int = 4, save_labeled: bool = False):
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        logger.info(f"Using device: {self.device}")
        self.frame_extractor = FrameExtractor(target_fps=target_fps, blur_threshold=blur_threshold)
        self.preprocessor = ImagePreprocessor(target_size=target_size, enable_clahe=enable_clahe)
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracker = TrackerAdapter(reid_weights, device=self.device)
        self.visualizer = Visualizer()
        self.batch_size = max(1, int(batch_size))
        self.save_labeled = save_labeled
        self._annotations = []#to store details of the annotation to json for user application to read

    def process_video(self, video_path: str, output_dir: str, save_video: bool=False,
                      save_comparison: bool=False, save_json: bool=False, min_track_length: int=3):
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("Cannot open video")
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        output_fps = self.frame_extractor.target_fps if self.frame_extractor.target_fps else source_fps
        video_writer_path = output_dir / f"{video_path.stem}_tracked.mp4" if save_video else None
        labeled_dir = output_dir / "labeled" if self.save_labeled else None
        if labeled_dir:
            labeled_dir.mkdir(parents=True, exist_ok=True)

        writer = AsyncWriter(video_path=video_writer_path, video_fps=output_fps, frame_size=(frame_w, frame_h), save_labeled_dir=labeled_dir)
        writer.start()

        inputs=[]; originals=[]; metas=[]; idxs=[]
        processed = 0; t0=time.time()
        for frame_idx, frame, metrics in self.frame_extractor.extract_frames(video_path, verbose=True):
            proc, meta = self.preprocessor.preprocess(frame, return_original=False)
            img_rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            inputs.append(img_rgb); originals.append(frame.copy()); metas.append(meta); idxs.append(frame_idx)
            if len(inputs) >= self.batch_size:
                self._run_batch(inputs, originals, metas, idxs, writer, save_comparison)
                processed += len(inputs)
                inputs.clear(); originals.clear(); metas.clear(); idxs.clear()
        if len(inputs)>0:
            self._run_batch(inputs, originals, metas, idxs, writer, save_comparison)
            processed += len(inputs)

        #finish
        writer.queue.join()
        writer.stop()
        writer.join()
        elapsed = time.time() - t0
        results = {
            'video': str(video_path),
            'processed_frames': processed,
            'elapsed_seconds': round(elapsed,2),
            'avg_fps': round(processed/elapsed,2) if elapsed>0 else None,
            'extractor_stats': dict(self.frame_extractor.stats),
            'annotations': self._annotations
        }
        if save_json:
            out_json = output_dir / f"{video_path.stem}_results.json"
            with open(out_json,'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results JSON: {out_json}")
        logger.info(f"Done: processed={processed} elapsed={elapsed:.1f}s avg_fps={results['avg_fps']}")
        return results

    def _run_batch(self, inputs, originals, metas, frame_idxs, writer: AsyncWriter, save_comparison=False):
        """
        Robust batch processing that always draws annotations via Visualizer class
        - inputs: list of RGB images (resized/padded to model input)
        - originals: list of original BGR frames (for visualization)
        - metas: list of metadata per image (scale, pad)
        - frame_idxs: list of frame indices
        """
        try:
            results = self.model.predict(source=inputs, imgsz=self.preprocessor.target_size, device=self.device, verbose=False)
        except Exception as e:
            logger.exception(f"Inference failed: {e}")
            return

        for res, orig, meta, fidx in zip(results, originals, metas, frame_idxs):
            dets = parse_ultralytics_result(res, conf_thresh=self.conf_threshold)
            dets_np = np.array(dets, dtype=np.float32) if len(dets) else np.empty((0,6), dtype=np.float32)

            try:
                tracks = self.tracker.update(dets_np, orig)
            except Exception as e:
                logger.warning(f"Tracker update failed for frame {fidx}: {e}")
                tracks = []

            # Normalize tracks into original-image coordinates and a canonical list for drawing
            tracks_orig = []
            for tr in tracks:
                # Accept various formats and fallback defensively
                try:
                    x1, y1, x2, y2, tid, cls, conf = tr
                    # Some trackers may already return coordinates in original image space.
                    # We assume values referring to model-space unless they exceed processed size; use metadata to detect.
                    # If bbox values are <= processed size (target_size), treat as model coords and transform back.
                    is_model_space = (max(x2, y2) <= max(self.preprocessor.target_size, self.preprocessor.target_size))
                    if is_model_space:
                        bbox_orig = self._transform_bbox([x1, y1, x2, y2], meta)
                        x1o = max(0, int(round(bbox_orig[0])))
                        y1o = max(0, int(round(bbox_orig[1])))
                        x2o = min(orig.shape[1]-1, int(round(bbox_orig[2])))
                        y2o = min(orig.shape[0]-1, int(round(bbox_orig[3])))
                    else:
                        # assume tracker already returned original-space coords
                        x1o = max(0, int(round(x1)))
                        y1o = max(0, int(round(y1)))
                        x2o = min(orig.shape[1]-1, int(round(x2)))
                        y2o = min(orig.shape[0]-1, int(round(y2)))
                    tracks_orig.append([x1o, y1o, x2o, y2o, int(tid) if tid is not None else -1, int(cls) if cls is not None else -1, float(conf) if conf is not None else 0.0])
                except Exception:
                    logger.debug(f"Skipping unexpected track format: {tr}")
                    continue

            # if no tracks, fall back to drawing detections
            if len(tracks_orig) == 0 and len(dets) > 0:
                # Transform detections to original space and draw as "untracked" with id=-1
                for det in dets:
                    mx1, my1, mx2, my2, mconf, mcls = det
                    bbox_orig = self._transform_bbox([mx1, my1, mx2, my2], meta)
                    x1o = max(0, int(round(bbox_orig[0]))); y1o = max(0, int(round(bbox_orig[1])))
                    x2o = min(orig.shape[1]-1, int(round(bbox_orig[2]))); y2o = min(orig.shape[0]-1, int(round(bbox_orig[3])))
                    tracks_orig.append([x1o, y1o, x2o, y2o, -1, int(mcls), float(mconf)])
            ann_entry = {
            'frame_idx': int(fidx),
            'annotations': []}
            for tr in tracks_orig:
                x1o, y1o, x2o, y2o, tid, cls, conf = tr
                ann_entry['annotations'].append({
                    'bbox': [int(x1o), int(y1o), int(x2o), int(y2o)],
                    'track_id': int(tid) if tid is not None else -1,
                    'class_id': int(cls) if cls is not None else -1,
                    'label': CLASS_NAMES.get(int(cls), str(cls)) if cls is not None else None,
                    'confidence': float(conf) if conf is not None else 0.0
                })
            self._annotations.append(ann_entry)

            annotated_frame = self.visualizer.draw_tracks(orig.copy(), tracks_orig)
            if writer.writer is not None:
                ok = writer.put({'type': 'frame', 'frame': annotated_frame})
                if not ok:
                    logger.debug("Dropped annotated frame (writer queue full)")

            #save labelled images here
            if self.save_labeled and writer.save_labeled_dir:
                ok = writer.put({'type': 'labeled', 'frame_idx': fidx, 'tracks': tracks_orig, 'frame': annotated_frame}, block=False)
                if not ok:
                    logger.debug("Dropped labeled item (writer queue full)")

            if save_comparison:
                enhanced = self.preprocessor.apply_clahe(self.preprocessor.resize_image(orig)[0]) if self.preprocessor.enable_clahe else self.preprocessor.resize_image(orig)[0]
                enhanced_up = cv2.resize(enhanced, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
                comp = self.visualizer.create_comparison_view(orig.copy(), enhanced_up, tracks_orig)
                if writer.writer is not None:
                    writer.put({'type': 'frame', 'frame': comp})

    def _transform_bbox(self, bbox, metadata):
        x1,y1,x2,y2 = bbox
        scale = metadata['scale']; pad_w, pad_h = metadata['pad']
        x1 = (x1 - pad_w) / scale; y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale; y2 = (y2 - pad_h) / scale
        return [x1,y1,x2,y2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", default="results")
    p.add_argument("--target-fps", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--clahe", action="store_true")
    p.add_argument("--no-clahe", action="store_false", dest="clahe")
    p.set_defaults(clahe=True)
    p.add_argument("--blur-threshold", type=float, default=100.0)
    p.add_argument("--min-track-len", type=int, default=5)
    p.add_argument("--device", default="auto")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--save-json", action="store_true")
    p.add_argument("--save-labeled", action="store_true")
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = RoadDefectPipeline(
        model_path=args.model,
        reid_weights=None,
        target_fps=args.target_fps,
        target_size=args.imgsz,
        enable_clahe=args.clahe,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        blur_threshold=args.blur_threshold,
        device=args.device,
        batch_size=args.batch_size,
        save_labeled=args.save_labeled
    )
    results = pipeline.process_video(
        video_path=args.video,
        output_dir=str(out_dir),
        save_video=args.save_video,
        save_comparison=False,
        save_json=args.save_json,
        min_track_length=args.min_track_len
    )
    if args.save_json and results:
        logger.info("Saved JSON results")

if __name__ == "__main__":
    main()