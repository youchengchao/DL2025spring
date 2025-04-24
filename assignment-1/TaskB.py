import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi-Head Self-Attention
import torch
import torch.nn as nn

# Multi-Head Self-Attention
class MHSA(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        out, _ = self.attn(x, x, x)
        out = self.norm(out + x)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out

# 主模型 TaskBNet
class TaskBNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, dim=32):
        super().__init__()
        self.wide1 = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112x112
        )

        self.down_mhsa = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=False),  # 56x56
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28
            MHSA(dim),
        )

        self.wide2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.wide1(x)
        x = self.down_mhsa(x)
        x = self.wide2(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

# Ablation without MHSA
class Ablation_MHSA(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, dim=32):
        super().__init__()
        self.wide1 = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.wide2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.wide1(x)
        x = self.wide2(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

# Ablation without wide1
class Ablation_wide1(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, dim=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112x112
        )
        self.down_mhsa = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=False),  # 56x56
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28
            MHSA(dim),
        )
        self.wide2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.down_mhsa(x)
        x = self.wide2(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

# Ablation without wide2
class Ablation_wide2(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, dim=32):
        super().__init__()
        self.wide1 = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112x112
        )
        self.down_mhsa = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=False),  # 56x56
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            MHSA(dim),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.wide1(x)
        x = self.down_mhsa(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)
