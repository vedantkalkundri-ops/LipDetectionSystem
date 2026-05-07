import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.model import LipPhraseNet
from ml.phrase_config import class_count


def main():
    parser = argparse.ArgumentParser(description="Export trained lip phrase model to ONNX.")
    parser.add_argument("--checkpoint", default="models/lip_phrase_baseline.pt")
    parser.add_argument("--config", default="config/phrases_50.json")
    parser.add_argument("--sequence-len", type=int, default=20)
    parser.add_argument("--out", default="models/lip_phrase_baseline.onnx")
    parser.add_argument("--max-classes", type=int, default=50, help="Export classifier size for labels 0..max_classes-1")
    args = parser.parse_args()

    num_classes = max(1, min(args.max_classes, class_count(args.config)))
    model = LipPhraseNet(num_classes=num_classes)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.randn(1, args.sequence_len, 1, 64, 64)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["frames"],
        output_names=["logits"],
        dynamic_axes={
            "frames": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=14,
    )
    print(f"ONNX exported -> {out_path}")


if __name__ == "__main__":
    main()
