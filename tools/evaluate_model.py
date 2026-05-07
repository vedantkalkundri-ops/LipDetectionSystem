import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.dataset import LipClipDataset
from ml.model import LipPhraseNet
from ml.phrase_config import build_id_maps, class_count


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model and write confusion matrix report.")
    parser.add_argument("--checkpoint", default="models/lip_phrase_baseline.pt")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--config", default="config/phrases_50.json")
    parser.add_argument("--sequence-len", type=int, default=20)
    parser.add_argument("--report", default="models/eval_report.json")
    parser.add_argument("--max-classes", type=int, default=50, help="Evaluate using labels 0..max_classes-1")
    args = parser.parse_args()

    device = torch.device("cpu")
    max_classes = max(1, min(args.max_classes, class_count(args.config)))
    allowed = set(range(max_classes))
    dataset = LipClipDataset(Path(args.processed_dir) / "test", sequence_len=args.sequence_len, allowed_labels=allowed)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    num_classes = max_classes
    model = LipPhraseNet(num_classes=num_classes).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            for gt, pd in zip(y.tolist(), pred.tolist()):
                confusion[gt, pd] += 1

    top1 = (correct / total) if total else 0.0
    id_to_key, _, _ = build_id_maps(args.config)
    hardest = []
    for gt in range(num_classes):
        row = confusion[gt]
        row_wo_diag = row.copy()
        row_wo_diag[gt] = 0
        if row_wo_diag.sum() == 0:
            continue
        pd = int(row_wo_diag.argmax())
        hardest.append(
            {
                "target": id_to_key.get(gt, str(gt)),
                "confused_with": id_to_key.get(pd, str(pd)),
                "count": int(row_wo_diag[pd]),
            }
        )
    hardest.sort(key=lambda x: x["count"], reverse=True)

    report = {
        "top1_accuracy": top1,
        "samples": total,
        "num_classes": num_classes,
        "hard_confusions": hardest[:20],
        "confusion_matrix": confusion.tolist(),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Top1 accuracy: {top1:.4f}")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
