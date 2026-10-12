from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ZONE_NORMALIZED = np.array(
    [[0.28, 0.44], [0.94, 0.44], [0.98, 0.99], [0.10, 0.99]], dtype=np.float32
)


@dataclass
class TrackState:
    inside_frames: int = 0
    alerted: bool = False
    last_seen_frame: int = 0


@dataclass
class ClipSummary:
    filename: str
    width: int
    height: int
    fps: float
    frames: int
    duration_seconds: float
    detected_frames: int = 0
    zone_positive_frames: int = 0
    events: int = 0
    inference_ms: list[float] = field(default_factory=list)
    end_to_end_ms: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "filename": self.filename,
            "resolution": f"{self.width}x{self.height}",
            "fps": round(self.fps, 3),
            "frames": self.frames,
            "duration_seconds": round(self.duration_seconds, 3),
            "detected_frames": self.detected_frames,
            "zone_positive_frames": self.zone_positive_frames,
            "events": self.events,
            "mean_inference_ms": round(float(np.mean(self.inference_ms)), 2),
            "median_inference_ms": round(float(np.median(self.inference_ms)), 2),
            "p95_inference_ms": round(float(np.percentile(self.inference_ms, 95)), 2),
            "mean_end_to_end_ms": round(float(np.mean(self.end_to_end_ms)), 2),
            "p95_end_to_end_ms": round(float(np.percentile(self.end_to_end_ms, 95)), 2),
        }


def point_inside(point: tuple[int, int], polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon.astype(np.int32), point, False) >= 0


def draw_label(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (x, y - th - 12), (x + tw + 12, y + 4), color, -1)
    cv2.putText(frame, text, (x + 6, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def process_video(model_path: Path, video_path: Path, out_dir: Path, conf: float, dwell_seconds: float) -> ClipSummary:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    summary = ClipSummary(video_path.name, width, height, fps, total, total / fps)

    annotated_dir = out_dir / "annotated_videos"
    alerts_dir = out_dir / "alerts"
    samples_dir = out_dir / "samples"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    alerts_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    output_w = 720
    output_h = round(height * output_w / width)
    writer = cv2.VideoWriter(
        str(annotated_dir / f"{video_path.stem}_annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_w, output_h),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create output video")

    polygon = (ZONE_NORMALIZED * np.array([width, height], dtype=np.float32)).astype(np.int32)
    dwell_frames = max(1, round(dwell_seconds * fps))
    states: dict[int, TrackState] = {}
    model = YOLO(str(model_path))
    event_rows: list[dict] = []
    frame_index = 0
    sample_indices = {round(total * 0.35), round(total * 0.55), round(total * 0.75)}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        result = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=conf,
            imgsz=640,
            device="cpu",
            verbose=False,
        )[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        summary.end_to_end_ms.append(elapsed_ms)
        summary.inference_ms.append(float(result.speed.get("inference", elapsed_ms)))

        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], (255, 170, 35))
        frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
        cv2.polylines(frame, [polygon], True, (255, 205, 60), 6, cv2.LINE_AA)
        draw_label(frame, "RESTRICTED ZONE", (max(20, int(polygon[0][0])), max(50, int(polygon[0][1]) - 10)), (45, 116, 173))

        frame_has_detection = False
        frame_has_inside = False
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            ids = result.boxes.id
            track_ids = ids.int().cpu().tolist() if ids is not None else list(range(len(boxes)))
            frame_has_detection = True

            for box, confidence, track_id in zip(boxes, confs, track_ids):
                x1, y1, x2, y2 = map(int, box)
                foot = (int((x1 + x2) / 2), int(y2))
                inside = point_inside(foot, polygon)
                frame_has_inside = frame_has_inside or inside
                state = states.setdefault(int(track_id), TrackState())
                state.last_seen_frame = frame_index
                if inside:
                    state.inside_frames += 1
                else:
                    state.inside_frames = 0
                    state.alerted = False

                confirmed = state.inside_frames >= dwell_frames
                color = (45, 45, 230) if confirmed else ((0, 170, 255) if inside else (60, 220, 130))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
                cv2.circle(frame, foot, 10, color, -1)
                status = "ALERT" if confirmed else ("IN ZONE" if inside else "person")
                draw_label(frame, f"ID {track_id}  {status}  {confidence:.2f}", (x1, max(42, y1)), color)

                if confirmed and not state.alerted:
                    state.alerted = True
                    summary.events += 1
                    event_time = frame_index / fps
                    alert_name = f"{video_path.stem}_event_{summary.events:02d}.jpg"
                    cv2.imwrite(str(alerts_dir / alert_name), frame)
                    event_rows.append(
                        {
                            "video": video_path.name,
                            "event_number": summary.events,
                            "track_id": int(track_id),
                            "frame": frame_index,
                            "timestamp_seconds": round(event_time, 3),
                            "confidence": round(float(confidence), 4),
                            "alert_image": alert_name,
                        }
                    )

        if frame_has_detection:
            summary.detected_frames += 1
        if frame_has_inside:
            summary.zone_positive_frames += 1

        stale = [tid for tid, state in states.items() if frame_index - state.last_seen_frame > int(fps)]
        for tid in stale:
            del states[tid]

        header = f"{video_path.name}   frame {frame_index}/{total}   events {summary.events}"
        draw_label(frame, header, (24, 54), (16, 38, 54))
        if frame_index in sample_indices:
            cv2.imwrite(str(samples_dir / f"{video_path.stem}_frame_{frame_index:05d}.jpg"), frame)
        writer.write(cv2.resize(frame, (output_w, output_h), interpolation=cv2.INTER_AREA))
        frame_index += 1

    cap.release()
    writer.release()

    if event_rows:
        csv_path = out_dir / "events.csv"
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as stream:
            writer_csv = csv.DictWriter(stream, fieldnames=event_rows[0].keys())
            if write_header:
                writer_csv.writeheader()
            writer_csv.writerows(event_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run restricted-area person monitoring on hallway videos.")
    parser.add_argument("--input", default="input_videos", help="Folder containing .mov or .mp4 files")
    parser.add_argument("--output", default="results", help="Output folder")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model weights")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--dwell", type=float, default=0.5, help="Seconds inside zone before alert")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_csv = output_dir / "events.csv"
    if events_csv.exists():
        events_csv.unlink()

    video_paths = sorted([*input_dir.glob("*.mov"), *input_dir.glob("*.mp4")])
    if not video_paths:
        raise SystemExit(f"No videos found in {input_dir}")
    summaries = [process_video(Path(args.model), video, output_dir, args.confidence, args.dwell) for video in video_paths]
    payload = {
        "model": Path(args.model).name,
        "person_class_only": True,
        "confidence_threshold": args.confidence,
        "dwell_seconds": args.dwell,
        "zone_normalized": ZONE_NORMALIZED.tolist(),
        "clips": [s.as_dict() for s in summaries],
    }
    (output_dir / "run_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
