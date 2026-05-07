from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _normalize_clip(frames: np.ndarray) -> np.ndarray:
    frames = frames.astype(np.float32) / 255.0
    mean = frames.mean()
    std = frames.std() + 1e-6
    return (frames - mean) / std


def _pad_or_trim(frames: np.ndarray, target_len: int) -> np.ndarray:
    n = frames.shape[0]
    if n == target_len:
        return frames
    if n > target_len:
        return frames[:target_len]
    pad_count = target_len - n
    pad = np.repeat(frames[-1][None, ...], pad_count, axis=0)
    return np.concatenate([frames, pad], axis=0)


class LipClipDataset(Dataset):
    def __init__(self, split_dir: str, sequence_len: int = 20, allowed_labels: Optional[Set[int]] = None):
        self.sequence_len = sequence_len
        all_files: List[Path] = sorted(Path(split_dir).glob("*.npz"))
        self.allowed_labels = allowed_labels
        if allowed_labels is None:
            self.files = all_files
        else:
            filtered: List[Path] = []
            for path in all_files:
                payload = np.load(path)
                label = int(payload["label"])
                if label in allowed_labels:
                    filtered.append(path)
            self.files = filtered
        if not self.files:
            raise RuntimeError(f"No .npz files found in {split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.files[idx]
        payload = np.load(path)
        frames = payload["frames"]  # [T, H, W]
        label = int(payload["label"])
        frames = _pad_or_trim(frames, self.sequence_len)
        frames = _normalize_clip(frames)
        frames = np.expand_dims(frames, axis=1)  # [T, 1, H, W]
        x = torch.tensor(frames, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        return x, y
