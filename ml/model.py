import torch
import torch.nn as nn


class FrameEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Linear(64 * 4 * 4, 128)

    def forward(self, x):
        # x: [B*T, 1, H, W]
        f = self.net(x)
        f = f.flatten(1)
        return self.proj(f)


class LipPhraseNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.encoder = FrameEncoder()
        self.temporal = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: [B, T, 1, H, W]
        bsz, timesteps = x.shape[:2]
        x = x.reshape(bsz * timesteps, *x.shape[2:])
        feat = self.encoder(x)
        feat = feat.reshape(bsz, timesteps, -1)
        seq_out, _ = self.temporal(feat)
        pooled = seq_out.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
