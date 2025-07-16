"""
OUAAZIZ MOUHCINE
Real-time Bacteria Detection and Tracking for Embedded Systems
Optimized for speed while maintaining small object detection capability
"""

import os
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict, deque
import json
from typing import List, Dict, Tuple
import argparse
from dataclasses import dataclass

VIDEO_PATH = "/home/mouaaziz/Desktop/PAalive_230615_1.avi"
MODEL_PATH = "/home/mouaaziz/yolo-distiller/runs/detect/MGD_a5e-5_λ0.7_exp/weights/best.pt"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realtime_detection_results")

CONFIDENCE_THRESHOLD = 0.2
NMS_IOU_THRESHOLD = 0.4
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

SLICE_HEIGHT = 640           
SLICE_WIDTH = 640            
OVERLAP_PIXELS = 100         
OVERLAP_RATIO = 100 / 640  

TARGET_FPS = 1

MAX_TRACK_AGE = 10
MIN_TRACK_HITS = 3
TRACK_IOU_THRESHOLD = 0.3

TORCH_COMPILE = True

@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    frame_id: int

@dataclass
class Track:
    track_id: int
    detections: deque
    last_detection: Detection
    age: int
    hits: int
    confirmed: bool

class EmbeddedBacteriaDetector:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model = self._load_optimized_model(model_path)
        self.tracks = {}
        self.next_track_id = 1
        self.frame_count = 0
        
        self.inference_times = deque(maxlen=100)
        self.detection_counts = deque(maxlen=100)
        
    def _load_optimized_model(self, model_path: str) -> YOLO:
        """Load and optimize YOLO model for embedded deployment."""
        print(f"Loading model for {self.device}...")
        model = YOLO(model_path)
        
        model.to(self.device)
        
        if TORCH_COMPILE and hasattr(torch, 'compile'):
            try:
                model.model = torch.compile(model.model, mode='max-autotune')
                print("Model compiled for optimization")
            except Exception as e:
                print(f"Could not compile model: {e}")
        
        self._warmup_model(model)
        
        return model
    
    def _warmup_model(self, model: YOLO):
        """Warmup model with dummy input."""
        print("Warming up model...")
        dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        with torch.no_grad():
            for _ in range(3):
                _ = model.predict(
                    source=dummy_input,
                    conf=CONFIDENCE_THRESHOLD,
                    iou=NMS_IOU_THRESHOLD,
                    device=self.device,
                    verbose=False,
                    int8=True
                )
        print("Model warmup completed")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Lightweight preprocessing for embedded systems."""
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        return frame
    
    def maximum_detection_slicing(self, frame: np.ndarray) -> List[Detection]:
        """Maximum detection using exact training configuration slicing."""
        print(f"  Using maximum detection slicing (frame {self.frame_count})")
        start_time = time.time()
        
        all_detections = []
        h, w = frame.shape[:2]
        
        step_h = SLICE_HEIGHT - OVERLAP_PIXELS
        step_w = SLICE_WIDTH - OVERLAP_PIXELS
        
        tiles_processed = 0
        
        for y in range(0, h, step_h):
            for x in range(0, w, step_w):
                y1 = y
                y2 = min(y + SLICE_HEIGHT, h)
                x1 = x  
                x2 = min(x + SLICE_WIDTH, w)
                
                if y2 - y1 < SLICE_HEIGHT and y2 == h:
                    y1 = max(0, h - SLICE_HEIGHT)
                if x2 - x1 < SLICE_WIDTH and x2 == w:
                    x1 = max(0, w - SLICE_WIDTH)
                
                tile = frame[y1:y2, x1:x2]
                
                if tile.shape[0] != SLICE_HEIGHT or tile.shape[1] != SLICE_WIDTH:
                    tile_padded = np.zeros((SLICE_HEIGHT, SLICE_WIDTH, 3), dtype=np.uint8)
                    tile_padded[:tile.shape[0], :tile.shape[1]] = tile
                    tile = tile_padded
                
                with torch.no_grad():
                    results = self.model.predict(
                        source=tile,
                        conf=CONFIDENCE_THRESHOLD,
                        iou=NMS_IOU_THRESHOLD,
                        device=self.device,
                        verbose=False,
                        int8= True
                    )
                
                tiles_processed += 1
                
                if len(results[0].boxes) > 0:
                    tile_detections = self._results_to_detections(results[0], self.frame_count)
                    for det in tile_detections:
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = det.bbox
                        global_x1 = bbox_x1 + x1
                        global_y1 = bbox_y1 + y1
                        global_x2 = bbox_x2 + x1
                        global_y2 = bbox_y2 + y1
                        
                        global_x1 = max(0, min(global_x1, w))
                        global_y1 = max(0, min(global_y1, h))
                        global_x2 = max(0, min(global_x2, w))
                        global_y2 = max(0, min(global_y2, h))
                        
                        det.bbox = (global_x1, global_y1, global_x2, global_y2)
                        all_detections.append(det)
        
        print(f"    Processed {tiles_processed} tiles")
        print(f"    Raw detections before NMS: {len(all_detections)}")
        
        merged_detections = self._comprehensive_nms(all_detections, NMS_IOU_THRESHOLD)
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.detection_counts.append(len(merged_detections))
        
        print(f"    Final detections after NMS: {len(merged_detections)} in {inference_time:.3f}s")
        return merged_detections
    
    def _comprehensive_nms(self, detections: List[Detection], iou_threshold: float = 0.4) -> List[Detection]:
        """Comprehensive NMS implementation for maximum detection accuracy."""
        if not detections:
            return []
        
        class_detections = defaultdict(list)
        for det in detections:
            class_detections[det.class_id].append(det)
        
        merged = []
        for class_id, dets in class_detections.items():
            if not dets:
                continue
                
            boxes = []
            scores = []
            for det in dets:
                boxes.append(det.bbox)
                scores.append(det.confidence)
            
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=self.device)
            scores_tensor = torch.tensor(scores, dtype=torch.float32, device=self.device)
            
            keep_indices = torch.ops.torchvision.nms(boxes_tensor, scores_tensor, iou_threshold)
            
            for idx in keep_indices:
                merged.append(dets[idx])
        
        return merged
    
    def _calculate_iou(self, box1: Tuple[float, float, float, float], 
                      box2: Tuple[float, float, float, float]) -> float:
        """Fast IoU calculation."""
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2
        
        xi1 = max(x1, x1_p)
        yi1 = max(y1, y1_p)
        xi2 = min(x2, x2_p)
        yi2 = min(y2, y2_p)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_p - x1_p) * (y2_p - y1_p)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _results_to_detections(self, result, frame_id: int) -> List[Detection]:
        """Convert YOLO results to Detection objects."""
        detections = []
        
        if len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for i in range(len(boxes)):
                detection = Detection(
                    bbox=tuple(boxes[i]),
                    confidence=float(confidences[i]),
                    class_id=int(classes[i]),
                    frame_id=frame_id
                )
                detections.append(detection)
        
        return detections
    
    def update_tracks(self, detections: List[Detection]) -> List[Track]:
        """Simple and fast tracking for embedded systems."""
        for track in self.tracks.values():
            track.age += 1
        
        matched_tracks = set()
        matched_detections = set()
        
        for track_id, track in self.tracks.items():
            if track.age > MAX_TRACK_AGE:
                continue
                
            best_iou = 0
            best_detection_idx = -1
            
            for i, detection in enumerate(detections):
                if i in matched_detections:
                    continue
                    
                iou = self._calculate_iou(track.last_detection.bbox, detection.bbox)
                if iou > TRACK_IOU_THRESHOLD and iou > best_iou:
                    best_iou = iou
                    best_detection_idx = i
            
            if best_detection_idx != -1:
                detection = detections[best_detection_idx]
                track.detections.append(detection)
                track.last_detection = detection
                track.hits += 1
                track.age = 0
                
                if track.hits >= MIN_TRACK_HITS:
                    track.confirmed = True
                
                matched_tracks.add(track_id)
                matched_detections.add(best_detection_idx)
        
        for i, detection in enumerate(detections):
            if i not in matched_detections:
                track = Track(
                    track_id=self.next_track_id,
                    detections=deque([detection], maxlen=30),
                    last_detection=detection,
                    age=0,
                    hits=1,
                    confirmed=False
                )
                self.tracks[self.next_track_id] = track
                self.next_track_id += 1
        
        self.tracks = {tid: track for tid, track in self.tracks.items() 
                      if track.age <= MAX_TRACK_AGE}
        
        return [track for track in self.tracks.values() if track.confirmed]
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[Detection], List[Track], Dict]:
        """Process a single frame with maximum detection and tracking."""
        self.frame_count += 1
        
        frame = self.preprocess_frame(frame)
        
        print(f"Frame {self.frame_count}: Running maximum detection slicing...")
        detections = self.maximum_detection_slicing(frame)
        
        tracks = self.update_tracks(detections)
        
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        avg_detections = np.mean(self.detection_counts) if self.detection_counts else 0
        current_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        metrics = {
            'frame_id': self.frame_count,
            'detections': len(detections),
            'tracks': len(tracks),
            'avg_inference_time': avg_inference_time,
            'current_fps': current_fps,
            'target_fps': TARGET_FPS,
            'using_max_detection': True
        }
        
        return detections, tracks, metrics


def visualize_frame(frame: np.ndarray, detections: List[Detection], 
                   tracks: List[Track], metrics: Dict) -> np.ndarray:
    """Visualize detections and tracks on frame."""
    vis_frame = frame.copy()
    
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.bbox)
        
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label = f"PA: {detection.confidence:.2f}"
        cv2.putText(vis_frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    for track in tracks:
        if len(track.detections) < 2:
            continue
            
        points = []
        for det in list(track.detections)[-10:]:
            x1, y1, x2, y2 = det.bbox
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            points.append(center)
        
        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(vis_frame, points[i-1], points[i], (255, 0, 0), 2)
        
        if points:
            cv2.putText(vis_frame, f"ID:{track.track_id}", points[-1], 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    y_offset = 30
    metrics_text = [
        f"Frame: {metrics['frame_id']}",
        f"Detections: {metrics['detections']}",
        f"Tracks: {metrics['tracks']}",
        f"FPS: {metrics['current_fps']:.1f}/{metrics['target_fps']}",
        f"Inference: {metrics['avg_inference_time']*1000:.1f}ms",
        f"Max Detection: {metrics['using_max_detection']}"
    ]
    
    for text in metrics_text:
        cv2.putText(vis_frame, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
    
    return vis_frame


def main():
    """Main function for real-time bacteria detection."""
    parser = argparse.ArgumentParser(description='Real-time Bacteria Detection')
    parser.add_argument('--video', default=VIDEO_PATH, help='Path to video file')
    parser.add_argument('--model', default=MODEL_PATH, help='Path to YOLO model')
    parser.add_argument('--output', default=OUTPUT_DIR, help='Output directory')
    parser.add_argument('--device', default=DEVICE, help='Device to use (cpu/cuda)')
    parser.add_argument('--save-video', action='store_true', help='Save output video')
    parser.add_argument('--display', action='store_true', help='Display real-time results')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("="*60)
    print("REAL-TIME BACTERIA DETECTION FOR EMBEDDED SYSTEMS")
    print("="*60)
    print(f"Video: {args.video}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Target FPS: {TARGET_FPS}")
    print(f"Slice Configuration: {SLICE_HEIGHT}x{SLICE_WIDTH} with {OVERLAP_PIXELS}px overlap")
    print()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    detector = EmbeddedBacteriaDetector(args.model, args.device)
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video: {args.video}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
    
    video_writer = None
    if args.save_video:
        output_path = os.path.join(args.output, 'gpu_bacteria_detection_output_optimize.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output video to: {output_path}")
    
    all_metrics = []
    start_time = time.time()
    frame_times = []
    
    try:
        frame_idx = 0
        while True:
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            detections, tracks, metrics = detector.process_frame(frame)
            
            vis_frame = visualize_frame(frame, detections, tracks, metrics)
            
            all_metrics.append(metrics)
            frame_times.append(time.time() - frame_start)
            
            if video_writer is not None:
                video_writer.write(vis_frame)
            
            if args.display:
                cv2.imshow('Real-time Bacteria Detection', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_idx += 1
            
            if frame_idx > 1000:
                print("Stopping after 1000 frames for testing...")
                break
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if args.display:
            cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        generate_performance_report(all_metrics, total_time, args.output)


def generate_performance_report(metrics_list: List[Dict], total_time: float, output_dir: str):
    """Generate performance analysis report."""
    if not metrics_list:
        return
    
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS REPORT")
    print("="*60)
    
    total_frames = len(metrics_list)
    total_detections = sum(m['detections'] for m in metrics_list)
    total_tracks = len(set().union(*[set() for m in metrics_list]))
    
    inference_times = [m['avg_inference_time'] for m in metrics_list if m['avg_inference_time'] > 0]
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    avg_fps = np.mean([m['current_fps'] for m in metrics_list if m['current_fps'] > 0])
    
    max_detection_usage = sum(1 for m in metrics_list if m.get('using_max_detection', False))
    
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Total frames processed: {total_frames}")
    print(f"Average processing FPS: {total_frames/total_time:.1f}")
    print(f"Target FPS: {TARGET_FPS}")
    print(f"Real-time capability: {'✓' if total_frames/total_time >= TARGET_FPS else '✗'}")
    print()
    print(f"DETECTION PERFORMANCE:")
    print(f"  Total detections: {total_detections}")
    print(f"  Average detections per frame: {total_detections/total_frames:.1f}")
    print(f"  Average inference time: {avg_inference_time*1000:.1f}ms")
    print(f"  Average inference FPS: {avg_fps:.1f}")
    print()
    print(f"MAXIMUM DETECTION SLICING:")
    print(f"  Slice size: {SLICE_HEIGHT}x{SLICE_WIDTH} pixels")
    print(f"  Overlap: {OVERLAP_PIXELS} pixels ({OVERLAP_RATIO*100:.1f}%)")
    print(f"  Maximum detection used on {max_detection_usage} frames ({max_detection_usage/total_frames*100:.1f}%)")
    print()
    
    import json
    metrics_file = os.path.join(output_dir, 'gpu_performance_metrics_optimize.json')
    with open(metrics_file, 'w') as f:
        json.dump({
            'summary': {
                'total_time': total_time,
                'total_frames': total_frames,
                'avg_fps': total_frames/total_time,
                'target_fps': TARGET_FPS,
                'total_detections': total_detections,
                'avg_inference_time': avg_inference_time,
                'max_detection_usage_percent': max_detection_usage/total_frames*100
            },
            'frame_metrics': metrics_list
        }, f, indent=2)
    
    print(f"Detailed metrics saved to: {metrics_file}")
    
    if total_frames/total_time >= TARGET_FPS:
        print("SYSTEM IS REAL-TIME CAPABLE for embedded deployment")
    else:
        print("System needs optimization for real-time performance")
        print("Consider: Lower resolution, simpler model, or hardware upgrade")


if __name__ == "__main__":
    main()
