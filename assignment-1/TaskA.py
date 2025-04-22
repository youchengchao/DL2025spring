import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicWeightConv(nn.Module):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, latent_dim=64):
        super().__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.latent_dim = latent_dim

        # 核心 kernel generator，不使用固定 in_channels，而是用 latent 向量做中繼表示
        self.kernel_generator = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU()
        )

        self.final_proj = None

    def forward(self, x):
        B, C, H, W = x.shape  # B: batch size, C: input channels
        in_channels = C

        # 第一次 forward 時根據實際通道數建立 final projection 層
        if self.final_proj is None:
            self.final_proj = nn.Linear(
                self.latent_dim, 
                self.out_channels * in_channels * self.kernel_size[0] * self.kernel_size[1]
            ).to(x.device)

        # 從輸入影像中取得每張圖的通道摘要 (mean pooling)，shape: (B, C)
        kernel_input = F.adaptive_avg_pool2d(x, (1, 1)).view(B, C)

        # 若輸入通道數不等於 latent_dim，需做一次線性轉換（動態建立 Linear）
        if C != self.latent_dim:
            # ⚠ 注意：這樣每次 forward 都會重新 new Linear，可考慮搬到 __init__ 搭配 ModuleDict 優化
            kernel_input = nn.Linear(C, self.latent_dim).to(x.device)(kernel_input)

        # 通過 kernel generator 建立中間特徵表示
        features = self.kernel_generator(kernel_input)

        # 通過 final projection 映射成 kernel 向量（每張圖一組 kernel）
        kernels = self.final_proj(features)

        # reshape 成 [B, Cout, Cin, kH, kW]
        kernels = kernels.view(B, self.out_channels, in_channels, *self.kernel_size)

        # 每張圖片用自己專屬的 kernel 做卷積
        outputs = []
        for i in range(B):
            out = F.conv2d(
                x[i:i+1],           # 單張圖片 (1, C, H, W)
                kernels[i],         # 對應的 kernel (Cout, Cin, kH, kW)
                stride=self.stride,
                padding=self.padding
            )
            outputs.append(out)

        # 把每張圖卷積結果接回一起，shape: (B, Cout, H_out, W_out)
        return torch.cat(outputs, dim=0)

class TaskA_AlexNet(nn.Module):
    def __init__(self, num_classes=1000):  # 可改成自己的類別數
        super().__init__()
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  
            nn.BatchNorm2d(96),
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
        print("初始化完成")

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) 
        x = self.classifier(x)
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)