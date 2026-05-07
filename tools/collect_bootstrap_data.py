import argparse
import csv
import json
import time
from pathlib import Path

import cv2


def load_phrases(config_path: Path):
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["phrases"]


def ensure_metadata_header(metadata_path: Path):
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        return
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_id",
                "speaker_id",
                "phrase_id",
                "phrase_key",
                "language",
                "video_path",
                "split",
                "duration_sec",
                "fps",
                "width",
                "height",
                "quality_ok",
                "notes",
            ]
        )


def append_metadata_row(metadata_path: Path, row):
    with metadata_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def record_clip(cap, out_path: Path, duration_s: float, fps: int, frame_size):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    start = time.time()
    while time.time() - start < duration_s:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        writer.write(frame)
        remaining = max(0.0, duration_s - (time.time() - start))
        preview = frame.copy()
        cv2.putText(
            preview,
            f"Recording... {remaining:0.1f}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Silent Speech Recorder", preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    writer.release()


def preview_countdown(cap, seconds: int = 2):
    start = time.time()
    while time.time() - start < seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        remaining = max(0.0, seconds - (time.time() - start))
        cv2.putText(
            frame,
            f"Starting in {remaining:0.1f}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Silent Speech Recorder", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Collect bootstrap phrase videos with webcam.")
    parser.add_argument("--speaker-id", required=True, help="Unique speaker identifier, e.g. s01.")
    parser.add_argument("--language", default="en", choices=["en", "hi", "es"])
    parser.add_argument("--clips-per-phrase", type=int, default=3)
    parser.add_argument("--duration-sec", type=float, default=2.5)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--config", default="config/phrases_50.json")
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--raw-dir", default="data/raw")
    args = parser.parse_args()

    config_path = Path(args.config)
    metadata_path = Path(args.metadata)
    raw_dir = Path(args.raw_dir)

    phrases = load_phrases(config_path)
    ensure_metadata_header(metadata_path)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera index and permissions.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    frame_size = (width, height)

    print("Recording uses auto-countdown. Press q in preview window to stop early.")
    sample_counter = int(time.time())

    try:
        for phrase in phrases:
            phrase_id = phrase["id"]
            phrase_key = phrase["key"]
            phrase_text = phrase["display"][args.language]
            print(f"\nPhrase {phrase_id:02d}: {phrase_text}")
            for clip_idx in range(args.clips_per_phrase):
                print(f"  Clip {clip_idx + 1}/{args.clips_per_phrase}: preparing...")
                should_continue = preview_countdown(cap, seconds=2)
                if not should_continue:
                    print("Stopped by user.")
                    return
                sample_id = f"{args.speaker_id}_{phrase_id:02d}_{sample_counter}"
                sample_counter += 1
                out_path = raw_dir / args.speaker_id / f"{phrase_id:02d}" / f"{sample_id}.mp4"
                record_clip(cap, out_path, args.duration_sec, args.fps, frame_size)
                append_metadata_row(
                    metadata_path,
                    [
                        sample_id,
                        args.speaker_id,
                        phrase_id,
                        phrase_key,
                        args.language,
                        str(out_path).replace("\\", "/"),
                        "",
                        args.duration_sec,
                        args.fps,
                        width,
                        height,
                        1,
                        "",
                    ],
                )
                print(f"  Saved: {out_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Collection complete.")


if __name__ == "__main__":
    main()
