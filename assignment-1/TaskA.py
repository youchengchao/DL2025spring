import torch
import torch.nn as nn

"""
class A_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_out_size=(1, 1), drop=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.pool_out_size = pool_out_size
        self.drop = drop  # 控制是否啟用 drop channel 機制

        kH, _ = self.kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=kH // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(pool_out_size)

    def pad_to_in_channels(self, x):
        B, C, H, W = x.shape
        if C < self.in_channels:
            pad = torch.zeros(B, self.in_channels - C, H, W, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        return x[:, :self.in_channels]

    def channel_dropout(self, x):
        B, C, H, W = x.shape
        mask = torch.ones_like(x)
        for i in range(B):
            drop_n = random.randint(1, C - 1) if C > 1 else 0
            drop_idx = torch.randperm(C)[:drop_n]
            mask[i, drop_idx] = 0
        return x * mask

    def forward(self, x):
        # training mode時，開啟dropout
        x = self.pad_to_in_channels(x)
        if self.drop and self.training:
            x = self.channel_dropout(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x
"""

class ChannelGate(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        gate = self.fc(x)  # shape: (B, C)
        return x * gate.view(x.size(0), -1, 1, 1)

class A_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_out_size=(1, 1), drop=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.pool_out_size = pool_out_size
        self.drop = drop  # 可選開啟 channel dropout

        self.channel_gate = ChannelGate(in_channels)

        kH, _ = self.kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=kH // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(pool_out_size)

    def pad_to_in_channels(self, x):
        B, C, H, W = x.shape
        if C < self.in_channels:
            pad = torch.zeros(B, self.in_channels - C, H, W, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        return x[:, :self.in_channels]

    def channel_dropout(self, x):
        B, C, H, W = x.shape
        mask = torch.ones_like(x)
        for i in range(B):
            drop_n = random.randint(1, C - 1) if C > 1 else 0
            drop_idx = torch.randperm(C)[:drop_n]
            mask[i, drop_idx] = 0
        return x * mask

    def forward(self, x):
        
        x = self.pad_to_in_channels(x)

        if self.drop and self.training:
            x = self.channel_dropout(x)
        
        gate = self.channel_gate(x) 
        x = x * gate

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class TaskA_AlexNet(nn.Module):
    def __init__(self, num_classes=1000):  # 可改成自己的類別數
        super().__init__()
        self.drop = None
        self.features = nn.Sequential(
            # conv1
            A_Conv(in_channels=3, out_channels=96, kernel_size=11, pool_out_size=(55, 55), drop=self.drop),
            nn.ReLU(inplace=True),
            # pool1
            nn.MaxPool2d(kernel_size=3, stride=2),                  
            # conv2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),           
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # pool2
            nn.MaxPool2d(kernel_size=3, stride=2),
            # conv3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),          
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),          
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),          
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # pool5
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),  # flatten後輸入
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.init_weights()

    def forward(self, x):
        if self.training:
            self.drop = True
        else:
            self.drop = False
        x = self.features(x)
        x = torch.flatten(x, 1) 
        x = self.classifier(x)
        return x
    
    def init_weights(self):
        self.drop = True
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
