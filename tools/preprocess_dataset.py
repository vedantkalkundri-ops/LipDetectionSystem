import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


def _has_mediapipe_solutions() -> bool:
    return hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")


def read_metadata(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mouth_bbox(face_landmarks, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    lip_points = []
    for edge in mp.solutions.face_mesh.FACEMESH_LIPS:
        idx = edge[0]
        lm = face_landmarks.landmark[idx]
        lip_points.append((int(lm.x * width), int(lm.y * height)))
    if not lip_points:
        return None
    xs = [p[0] for p in lip_points]
    ys = [p[1] for p in lip_points]
    pad_x = max(8, int(0.15 * (max(xs) - min(xs) + 1)))
    pad_y = max(8, int(0.30 * (max(ys) - min(ys) + 1)))
    x1 = max(0, min(xs) - pad_x)
    y1 = max(0, min(ys) - pad_y)
    x2 = min(width - 1, max(xs) + pad_x)
    y2 = min(height - 1, max(ys) + pad_y)
    return x1, y1, x2, y2


def extract_lip_clip(video_path: Path, target_size: int, max_frames: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    use_face_mesh = _has_mediapipe_solutions()
    face_mesh = None
    if use_face_mesh:
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
        )
    frames = []
    try:
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if use_face_mesh and face_mesh is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)
                if not result.multi_face_landmarks:
                    continue
                bbox = mouth_bbox(result.multi_face_landmarks[0], w, h)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
            else:
                # Fallback crop when mediapipe FaceMesh API is unavailable.
                x1 = int(0.25 * w)
                x2 = int(0.75 * w)
                y1 = int(0.58 * h)
                y2 = int(0.92 * h)
            roi = frame[y1 : y2 + 1, x1 : x2 + 1]
            if roi.size == 0:
                continue
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(roi, (target_size, target_size), interpolation=cv2.INTER_AREA)
            frames.append(roi)
    finally:
        cap.release()
        if face_mesh is not None:
            face_mesh.close()

    if len(frames) < 6:
        return None
    return np.stack(frames, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw videos into lip ROI .npz files.")
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--default-split", default="train", choices=["train", "val", "test"])
    args = parser.parse_args()

    metadata = read_metadata(Path(args.metadata))
    out_dir = Path(args.out_dir)
    for split in ("train", "val", "test"):
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    for row in metadata:
        video_path = Path(row["video_path"])
        if not video_path.exists():
            skipped += 1
            continue
        split = row["split"].strip() or args.default_split
        label = int(row["phrase_id"])
        sample_id = row["sample_id"]
        frames = extract_lip_clip(video_path, target_size=args.target_size, max_frames=args.max_frames)
        if frames is None:
            skipped += 1
            continue
        np.savez_compressed(out_dir / split / f"{sample_id}.npz", frames=frames, label=label, sample_id=sample_id)
        processed += 1

    print(f"Processed clips: {processed}")
    print(f"Skipped clips: {skipped}")


if __name__ == "__main__":
    main()
