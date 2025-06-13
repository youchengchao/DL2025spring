from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch.nn as nn
import torch.nn.functional as F

class DLHW2Net(nn.Module):
    def __init__(self, num_seg_classes=21, num_det_classes=10, num_cls=10):
        super().__init__()
        
        # 1) Backbone (MobileNetV3-Small)
        weights = MobileNet_V3_Small_Weights.DEFAULT
        base = mobilenet_v3_small(weights=weights)
        self.backbone = base.features  # [B, C, H/32, W/32]
    
        # 2) Neck：通用的特徵縮減
        self.neck = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 3) Heads
        #   Stage1: segmentation (上採樣到原圖)
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, num_seg_classes, kernel_size=1),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)
        )
        #   Stage2: detection (簡易 anchor-free)
        self.det_head = nn.Sequential(
            nn.Conv2d(256, num_det_classes*4, kernel_size=1),  # bbox delta
            # 也可拆成兩支：class_logits & bbox_regressor
        )

        #   Stage3: classification
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_fc   = nn.Linear(256, num_cls)

    def forward(self, x, stage=3):
        x = self.backbone(x)
        x = self.neck(x)

        if stage == 1:
            return self.seg_head(x)
        elif stage == 2:
            # 輸出 shape [B, num_det*4, H', W']
            return self.det_head(x)
        else:  # stage == 3
            x = self.cls_pool(x).view(x.size(0), -1)
            return self.cls_fc(x)
