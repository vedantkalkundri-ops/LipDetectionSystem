import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.dataset import LipClipDataset
from ml.model import LipPhraseNet
from ml.phrase_config import class_count


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct_top1 = 0
    correct_top3 = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            top1 = logits.argmax(dim=1)
            top3 = torch.topk(logits, k=min(3, logits.shape[1]), dim=1).indices
            total += y.size(0)
            correct_top1 += (top1 == y).sum().item()
            correct_top3 += (top3 == y.unsqueeze(1)).any(dim=1).sum().item()
    if total == 0:
        return 0.0, 0.0
    return correct_top1 / total, correct_top3 / total


def main():
    parser = argparse.ArgumentParser(description="Train baseline lip phrase classifier on CPU.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--config", default="config/phrases_50.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--sequence-len", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--out", default="models/lip_phrase_baseline.pt")
    parser.add_argument("--max-classes", type=int, default=50, help="Train using labels 0..max_classes-1")
    args = parser.parse_args()

    device = torch.device("cpu")
    max_classes = max(1, min(args.max_classes, class_count(args.config)))
    allowed = set(range(max_classes))
    train_ds = LipClipDataset(Path(args.processed_dir) / "train", sequence_len=args.sequence_len, allowed_labels=allowed)
    val_ds = LipClipDataset(Path(args.processed_dir) / "val", sequence_len=args.sequence_len, allowed_labels=allowed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = LipPhraseNet(num_classes=max_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    stale_epochs = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        train_loss = running_loss / max(1, len(train_loader))
        val_top1, val_top3 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d}: loss={train_loss:.4f} val_top1={val_top1:.4f} val_top3={val_top3:.4f}")

        if val_top1 > best_val:
            best_val = val_top1
            stale_epochs = 0
            torch.save(
                {"model_state": model.state_dict(), "val_top1": best_val, "num_classes": max_classes},
                out_path,
            )
            print(f"Saved checkpoint -> {out_path}")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping triggered.")
                break

    print(f"Best validation top1: {best_val:.4f}")
    print(f"Trained classes: {max_classes}")


if __name__ == "__main__":
    main()
