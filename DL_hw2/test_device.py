import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Use {device}")
print(device.type=="cuda")

from scripts.models.DLHW2Net import DLHW2Net
model = DLHW2Net().to("cpu")
#print(model)

# device是string或是torch.device好像都沒差?

from scripts.utils import seg_closure,  det_closure, cls_closure
stage = 3
closure_fn = seg_closure if stage==1 else det_closure if stage==2 else cls_closure

print(closure_fn)

# 假設 yolo_v1_loss, loc_criterion, obj_criterion, cls_criterion 都已經定義或正確 import
from scripts.utils import yolo_v1_loss
def debug_yolo_loss():
    # 1) 參數設定
    B = 2                 # batch size
    S = 7                 # grid size
    C = 3                 # 類別數 (ex: 3)
    H_img, W_img = 224, 224
    num_classes = C
    λ_coord, λ_obj, λ_noobj, λ_cls = 5.0, 1.0, 0.5, 1.0

    # 2) 构造 det_map
    #    隨機輸出 (或 zeros) 都行，dtype 可以試 float32 / float16 看有沒有 mismatch
    det_map = torch.randn(B, 5 + C, S, S, device='cuda').to(dtype=torch.float16)

    # 3) 构造 targets (list of list-of-dicts)
    #    第一張圖有一個 box，第二張圖沒有 target
    targets = [
        [ { 'bbox': [ 50,  30, 100, 120],   # x, y, w, h (像素座標)
            'category_id': 2 } ],          # 1-based 類別編號
        []  # 第二張圖空 list
    ]

    # 4) 呼叫 loss
    loss = yolo_v1_loss(
        det_map=det_map,
        targets=targets,
        img_size=(H_img, W_img),
        num_classes=num_classes,
        λ_coord=λ_coord,
        λ_obj=λ_obj,
        λ_noobj=λ_noobj,
        λ_cls=λ_cls
    )

    # 5) 列印 debug 資訊
    print(f"det_map shape:      {det_map.shape}, dtype: {det_map.dtype}")
    print(f"num_classes:        {num_classes}")
    print(f"img_size:           {(H_img, W_img)}")
    print("targets:", targets)
    print("Computed loss:", loss.item())

if __name__ == "__main__":
    debug_yolo_loss()