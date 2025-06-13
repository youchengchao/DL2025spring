import torch
import torch.nn as nn
import torch.nn.functional as F


class WideBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super(WideBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.dropout(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out += self.shortcut(x)
        return out



class TaskBNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, dim=32):
        super().__init__()
        self.channel_wise_Conv = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.down_mhsa = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=False),  # 56x56
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28
            MHSA(dim),
        )

        self.wide2 = nn.Sequential(
            nn.Conv2d(dim, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
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
            nn.Conv2d(dim, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
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
        self.down_mhsa = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, stride=2, padding=1, bias=False),  # 56x56
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28
            MHSA(dim),
        )
        self.wide2 = nn.Sequential(
            nn.Conv2d(dim, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

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
