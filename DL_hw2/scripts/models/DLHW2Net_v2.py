from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch.nn as nn
import torch.nn.functional as F

class DLHW2Net(nn.Module):
    def __init__(self, num_seg_classes=21, num_det_classes=10, num_cls=10):
        super().__init__()
        # 1) Backbone
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = base.features    # [B,576,H/32,W/32]

        # 2) WideConv neck (2 layers)
        self.neck = nn.Sequential(
            nn.Conv2d(576, 384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )

        # 3) Unified head (3 layers, single branch)
        c_mid = 128
        c_out = (5 + num_det_classes) + num_seg_classes + num_cls
        self.head = nn.Sequential(
            nn.Conv2d(384, c_mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 1),
        )
        # slice indices
        self._det_end = 5 + num_det_classes
        self._seg_end = self._det_end + num_seg_classes

    def forward(self, x, stage="all"):
        h_in, w_in = x.shape[2:]
        feat = self.neck(self.backbone(x))       # [B,384,h/32,w/32]
        y = self.head(feat)                      # [B,c_out,h/32,w/32]

        det_map = y[:, :self._det_end]           # [B,5+C_det,h',w']
        seg_map = y[:, self._det_end:self._seg_end]  # [B,C_seg,h',w']
        cls_map = y[:, self._seg_end:]           # [B,C_cls,h',w']

        key = str(stage).lower()
        if key in {"1", "seg"}:
            return F.interpolate(seg_map, (h_in, w_in), mode='bilinear', align_corners=False)
        if key in {"2", "det"}:
            return det_map
        if key in {"3", "cls"}:
            return cls_map.mean(dim=(2,3))       # global avg
        # "all"
        return det_map, seg_map, cls_map.mean(dim=(2,3))
