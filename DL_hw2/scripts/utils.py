import os
import pathlib
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection, ImageFolder
import tempfile, json
tqdm = __import__('tqdm').tqdm
from pycocotools.cocoeval import COCOeval

# Detection Transform
class CocoTransformWrapper:
    def __init__(self, size=224):
        self.size = size
        self.resize = transforms.Resize((size, size), interpolation=Image.BILINEAR)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    def __call__(self, image, target):
        orig_w, orig_h = image.size
        image = self.resize(image)
        new_w, new_h = image.size
        sx, sy = new_w/orig_w, new_h/orig_h
        for obj in target:
            x,y,w,h = obj['bbox']
            x1,y1,x2,y2 = x, y, x+w, y+h
            x1, y1, x2, y2 = x1*sx, y1*sy, x2*sx, y2*sy
            obj['bbox'] = [x1, y1, x2, y2]
        image = self.normalize(self.to_tensor(image))
        return image, target

# Segmentation Transform
class VOCSegTransform:
    def __init__(self, size=224):
        self.size = (size,size)
    def __call__(self, image, mask):
        image = transforms.functional.resize(image, self.size, interpolation=Image.BILINEAR)
        image = transforms.functional.to_tensor(image)
        '''
        image = transforms.functional.normalize(
            transforms.functional.to_tensor(image),
            mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
        )
        '''
        mask = transforms.functional.resize(mask, self.size, interpolation=Image.NEAREST)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image, mask

# DataLoaders
class MiniVOCSegDataset(Dataset):
    def __init__(self, root, image_set="train", transform=None):
        self.items=[]
        base = pathlib.Path(root)
        ids = (base / 'ImageSets' / 'Segmentation' / f"{image_set}.txt").read_text().split()
        for i in ids:
            self.items.append((base/'JPEGImages'/f"{i}.jpg", base/'SegmentationClass'/f"{i}.png"))
        self.transform=transform
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        p_img, p_mask = self.items[idx]
        img = Image.open(p_img).convert('RGB')
        m = Image.open(p_mask)
        return self.transform(img,m) if self.transform else (img, m)

def detection_collate_fn(batch): return tuple(zip(*batch))

def get_loaders(task, batch_size, num_workers, data_root, device: torch.device):
    pin = (device.type == "cuda")
    root = pathlib.Path(data_root)
    if task=='seg':
        ds1 = MiniVOCSegDataset(root/'mini_voc_seg','train',VOCSegTransform())
        ds2 = MiniVOCSegDataset(root/'mini_voc_seg','val',VOCSegTransform())
        return {"train":DataLoader(dataset=ds1,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=pin),
                "val": DataLoader(dataset=ds2,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin)}
    if task=='det':
        ds1 = CocoDetection(root/'mini_coco_det'/'train', root/'mini_coco_det'/'annotations'/'instances_train.json', transforms=CocoTransformWrapper())
        ds2 = CocoDetection(root/'mini_coco_det'/'val', root/'mini_coco_det'/'annotations'/'instances_val.json', transforms=CocoTransformWrapper())
        return {"train":DataLoader(dataset=ds1,batch_size=batch_size,shuffle=True,num_workers=num_workers, collate_fn=detection_collate_fn,pin_memory=pin),
                "val": DataLoader(dataset=ds2,batch_size=batch_size,shuffle=False,num_workers=num_workers, collate_fn=detection_collate_fn,pin_memory=pin)}
    if task=='cls':
        tr = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
        ds1 = ImageFolder(root/'imagenette160'/'train', tr)
        ds2 = ImageFolder(root/'imagenette160'/'val', tr)
        return {"train":DataLoader(dataset=ds1,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=pin),
                "val": DataLoader(dataset=ds2,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin)}
    raise ValueError(f"Unknown task {task}")

def eval_seg(model, dataloader, device: torch.device):
    num_classes = 21
    model.eval()
    conf_matrix = torch.zeros(num_classes, num_classes, device=device, dtype=torch.int64)
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks  = masks.to(device)
            preds  = model(images,'seg').argmax(dim=1)
            valid = masks.view(-1) < num_classes
            inds  = num_classes * masks.view(-1)[valid] + preds.view(-1)[valid]
            bincount = torch.bincount(inds, minlength=num_classes**2)
            conf_matrix += bincount.reshape(num_classes, num_classes)
    intersection = torch.diag(conf_matrix)
    union = conf_matrix.sum(1) + conf_matrix.sum(0) - intersection
    return (intersection.float() / (union.float() + 1e-6)).mean().item()

import torch.nn.functional as F
from pycocotools.coco import COCO
def eval_det(model, dataloader, device, iou_type='bbox',
             conf_thres=0.5, nms_thres=0.5, stride=32):

    model.eval()
    results = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="EvalDet"):
            imgs = torch.stack([img.to(device) for img in images])
            det_map = model(imgs, 'det')            # [B, 5+C, Hf, Wf]
            B, _, Hf, Wf = det_map.shape

            # 1. 解析輸出 --------------------------------------------------
            box_xy   = torch.sigmoid(det_map[:, :2])    # cx, cy
            box_wh   = det_map[:, 2:4].exp()            # w, h
            obj_conf = torch.sigmoid(det_map[:, 4])     # objness
            cls_prob = F.softmax(det_map[:, 5:], dim=1) # [B,C,Hf,Wf]

            # 2. 生成網格 --------------------------------------------------
            gy, gx = torch.meshgrid(torch.arange(Hf, device=device),
                                    torch.arange(Wf, device=device),
                                    indexing='ij')
            gx = gx.unsqueeze(0).unsqueeze(0)  # [1,1,Hf,Wf]
            gy = gy.unsqueeze(0).unsqueeze(0)

            # 3. 還原到影像座標 --------------------------------------------
            xc = (gx + box_xy[:, 0:1]) * stride
            yc = (gy + box_xy[:, 1:2]) * stride
            w  = box_wh[:, 0:1] * stride
            h  = box_wh[:, 1:2] * stride

            # 4. 逐張影像組裝結果 ------------------------------------------
            for i in range(B):
                xi, yi = xc[i].view(-1), yc[i].view(-1)
                wi, hi =  w[i].view(-1),  h[i].view(-1)
                conf_i = obj_conf[i].view(-1)
                prob_i = cls_prob[i].permute(1, 2, 0).reshape(-1, cls_prob.size(1))
                scores_i, labels_i = (conf_i.unsqueeze(1) * prob_i).max(dim=1)

                keep = scores_i > conf_thres
                if not keep.any():
                    continue

                xi, yi, wi, hi = xi[keep], yi[keep], wi[keep], hi[keep]
                sc, lb = scores_i[keep], labels_i[keep]

                x1, y1 = xi - wi / 2, yi - hi / 2
                x2, y2 = xi + wi / 2, yi + hi / 2
                boxes  = torch.stack([x1, y1, x2, y2], dim=1).cpu()

                img_id = int(targets[i][0]['image_id'])   # ← 保證有標註時一定存在
                for b, s, l in zip(boxes.tolist(), sc.cpu().tolist(), lb.cpu().tolist()):
                    results.append({
                        "image_id":    img_id,
                        "category_id": int(l),
                        "bbox":        [b[0], b[1], b[2]-b[0], b[3]-b[1]],
                        "score":       float(s)
                    })
    if len(results)==0: 
        return 0.0
    # ---------------- COCO 評估 -------------------------------------------
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
        json.dump(results, f)
        res_file = f.name

    # 重新載入 GT，並補上缺欄位
    coco_gt = dataloader.dataset.coco
    for k, default in (('info', {}), ('licenses', [])):
        if k not in coco_gt.dataset:
            coco_gt.dataset[k] = default

    coco_dt   = coco_gt.loadRes(res_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate();  coco_eval.accumulate();  coco_eval.summarize()

    return float(coco_eval.stats[0])

def eval_cls(model, dataloader, device: torch.device):
    """
    Evaluate image classification accuracy.

    Args:
        model: 已加载至 eval 模式的 classification model。
        dataloader: 返回 (images, labels) 的 DataLoader。
        device: torch device。

    Returns:
        accuracy (float): Top-1 准确率。
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Eval Cls", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images,'cls')
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return correct / total if total > 0 else 0.0

# closure_fn
def seg_closure(images, masks, model, loss_func, device: torch.device):
    # images: Tensor [B, 3, H, W], masks: LongTensor [B, H, W]
    images = images.to(device, non_blocking=True)
    masks  = masks.to(device, non_blocking=True, dtype=torch.long)
    outputs = model(images, stage=1)  # [B, C_seg, H, W]
    return loss_func(outputs, masks)  # CrossEntropyLoss(ignore_index=…)

def det_closure(images, targets, model, loss_func, device: torch.device):
    # images: list of Tensors → stack to [B,3,H,W]
    imgs = torch.stack([img.to(device, non_blocking=True) for img in images])
    det_map = model(imgs, stage=2)   # [B, 5+C_det, S, S]
    B, _, S, _ = det_map.shape
    _, _, H_img, W_img = imgs.shape
    num_classes = det_map.shape[1] - 5
    # loss_func is yolo_v1_loss
    return loss_func(det_map,
                     targets,
                     img_size=(H_img, W_img),
                     num_classes=num_classes)

def cls_closure(images, labels, model, loss_func, device: torch.device):
    # images: Tensor [B,3,H,W], labels: LongTensor [B]
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    outputs = model(images, stage=3)  # [B, C_cls]
    return loss_func(outputs, labels)

# loss 定義
loc_criterion = torch.nn.SmoothL1Loss(reduction='sum')
obj_criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
cls_criterion = torch.nn.CrossEntropyLoss(reduction='sum')

def yolo_v1_loss(det_map, targets, img_size, num_classes,
                 λ_coord=5.0, λ_obj=1.0, λ_noobj=0.5, λ_cls=1.0):

    B, _, S, _ = det_map.shape
    H, W = img_size
    device = det_map.device
    dtype     = det_map.dtype

    box_pred = det_map[:, 0:4]        # [B, 4, S, S]
    obj_pred = det_map[:, 4]          # [B, S, S], logit
    cls_pred = det_map[:, 5:]         # [B, C, S, S], raw scores

    box_tgt = torch.zeros_like(box_pred)
    obj_tgt = torch.zeros_like(obj_pred)

    cls_labels = []   # list of (b_index, gy, gx, label)
    for b, ann in enumerate(targets):
        
        # 因為ann是list of dict
        boxes  = torch.tensor(
            [[o["bbox"][0], o["bbox"][1], o["bbox"][0] + o["bbox"][2], o["bbox"][1] + o["bbox"][3]] for o in ann],
            device=device, dtype=torch.float32
        )
        labels = torch.tensor(
            [o["category_id"] - 1 for o in ann],
            device=device, dtype=torch.long
        )

        if boxes.numel() == 0:
            continue

        # 1) 計算 cxcywh（0~1）
        cxcy = ((boxes[:, :2] + boxes[:, 2:]) / 2)
        wh   = (boxes[:, 2:] - boxes[:, :2])
        cxcy[:, 0] /= W; cxcy[:, 1] /= H
        wh  [:, 0] /= W; wh  [:, 1] /= H

        cxcy = cxcy.to(device=device, dtype=dtype)
        wh   = wh  .to(device=device, dtype=dtype)

        # 2) grid idx
        gx = (cxcy[:,0] * S).clamp(0, S - 1e-4).long()
        gy = (cxcy[:,1] * S).clamp(0, S - 1e-4).long()

        # 3) 過濾越界
        valid = (labels >= 0) & (labels < num_classes)
        valid &= (gx >= 0) & (gx < S) & (gy >= 0) & (gy < S)
        if not valid.any():
            continue
        labels = labels[valid]; gx = gx[valid]; gy = gy[valid]
        cxcy   = cxcy[valid];   wh = wh[valid]

        # 4) 填入 loc + obj + 收集分類
        obj_tgt[b, gy, gx] = 1.0
        box_tgt[b, 0, gy, gx] = cxcy[:,0]
        box_tgt[b, 1, gy, gx] = cxcy[:,1]
        box_tgt[b, 2, gy, gx] = wh  [:,0]
        box_tgt[b, 3, gy, gx] = wh  [:,1]

        for yi, xi, lab in zip(gy.tolist(), gx.tolist(), labels.tolist()):
            cls_labels.append((b, yi, xi, lab))

    # masks
    pos_mask = obj_tgt > 0                      # [B,S,S]
    neg_mask = ~pos_mask

    # 1) Localization loss (only pos cells)
    mask_loc = pos_mask.unsqueeze(1).expand_as(box_pred)
    loc_loss = loc_criterion(box_pred[mask_loc], box_tgt[mask_loc])

    # 2) Objectness loss
    obj_loss   = obj_criterion(obj_pred[pos_mask], obj_tgt[pos_mask])
    noobj_loss = obj_criterion(obj_pred[neg_mask], obj_tgt[neg_mask])

    # 3) Classification loss using CrossEntropy
    #    收集所有正樣本的位置與 label，然後對應取出 cls_pred
    if cls_labels:
        # idx tensors
        bs = torch.tensor([x[0] for x in cls_labels], device=device)
        ys = torch.tensor([x[1] for x in cls_labels], device=device)
        xs = torch.tensor([x[2] for x in cls_labels], device=device)
        labs = torch.tensor([x[3] for x in cls_labels], device=device)  # [P]
        
        pred_scores = cls_pred[bs, :, ys, xs]  # → [P, C]
        cls_loss = cls_criterion(pred_scores, labs)
    else:
        cls_loss = torch.tensor(0.0, device=device)

    # 總 loss
    total_loss = (
        λ_coord * loc_loss +
        λ_obj   * obj_loss +
        λ_noobj * noobj_loss +
        λ_cls   * cls_loss
    ) / max(1, B)

    return total_loss


from tqdm import tqdm
def train_one_epoch(model, stage, dataloader, loss_func, optimizer, device: torch.device, closure_fn=None):
    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if isinstance(optimizer, torch.optim.LBFGS):
        is_lbfgs = True
    else:
        is_lbfgs = False
        scaler = torch.GradScaler() # 用新的 GradScaler API --> 降精度換取多餘空間

    running_loss = 0.0
    total_items  = 0

    loop = tqdm(dataloader, desc=f"Train Stage {stage}", leave=False)
    for batch in loop:
        optimizer.zero_grad()

        # LBFGS optimizer：必須要用closure、不能用GradScaler降精度
        if is_lbfgs:
            
            def _lbfgs_closure():
                optimizer.zero_grad()
                loss = closure_fn(*batch, model, loss_func, device)
                loss.backward()
                return loss

            loss = optimizer.step(_lbfgs_closure)

            # 計算batch_size
            if stage == 2:
                images, _ = batch
                batch_size = len(images)
            else:
                images, _ = batch
                batch_size = images.size(0)

        # 非 LBFGS optimizer: 可以用GradScaler降精度
        else:
            optimizer.zero_grad()
            with torch.autocast(device.type):
                loss = closure_fn(*batch, model, loss_func, device)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 計算batch_size
            if stage == 2:
                images, _ = batch
                batch_size = len(images)
            else:
                images, _ = batch
                batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        total_items  += batch_size
        loop.set_postfix(loss=f"{loss.item():.4f}", lr=optimizer.param_groups[0]["lr"])

    return running_loss / total_items

import time
def format_duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    elif m > 0:
        return f"{m:02d}m{s:02d}s"
    else:
        return f"{s:02d}s"

from torch.optim import LBFGS
import pathlib
def train_(model,
           stage: int,
           train_loader,
           val_loader,
           eval_loaders: dict,
           loss_func,
           optimizer,
           scheduler,
           device: torch.device,
           save_dir: pathlib.Path,
           reco_code = "",
           init_lr: float = 1e-5,
           n_epoch: int = 120):
    if stage not in (1, 2, 3):
        raise ValueError("Invalid stage. Use 1, 2 or 3.")

    log = {
        "training_setting": {"stage": stage, "init_lr": init_lr, "n_epoch": n_epoch},
        "train_loss": [],     # 每 epoch 的 train loss
        "val_loss": [],       # 每 epoch 的 val loss
        "lr": [],             # 每 epoch 用的 lr
        "stage_time": 0,      # 整個 stage 花的秒數
        "best_val_loss": float('inf'),
        # 以下為 stage 完成後要記錄的 metrics
        "mIoU_base": None,
        "mAP_base": None,
        "Top1_base": None,
        "mIoU_drop": None,
        "mAP_drop": None
    }

    closure_fn = {1: seg_closure, 2: det_closure, 3: cls_closure}[stage]

    start_time = time.time()
    for epoch in range(1, n_epoch + 1):
        model.train()
        train_loss = train_one_epoch(
            model,
            stage=stage,
            dataloader=train_loader,
            loss_func=loss_func,
            optimizer=optimizer,
            device=device,
            closure_fn=closure_fn
        )
        log["train_loss"].append(train_loss)

        # 驗證 loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                # --- Stage 2 (Detection) 用 closure_fn 解包 images, targets ---
                if stage == 2 and closure_fn:
                    loss_tensor = closure_fn(*batch, model, loss_func, device)

                # --- Stage 1 & 3 (Seg / Cls) --- 
                else:
                    images, labels = batch
                    if closure_fn:
                        loss_tensor = closure_fn(images, labels, model, loss_func, device)
                    else:
                        images = images.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        outputs     = model(images, stage)
                        loss_tensor = loss_func(outputs, labels)
                val_losses.append(loss_tensor.item())
        val_loss = sum(val_losses) / len(val_losses)
        log["val_loss"].append(val_loss)

        if val_loss < log["best_val_loss"]:
            log["best_val_loss"] = val_loss
            torch.save(model.state_dict(), save_dir / f"{reco_code}_stage{stage}_best.pth")
            print("model saved")

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        log["lr"].append(lr)

        print(f"Epoch {epoch}/{n_epoch} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={lr:.2e}")

    log["stage_time"] = time.time() - start_time

    # stage訓練完畢後計算每個任務的指標
    model.load_state_dict(torch.load(save_dir / f"{reco_code}_stage{stage}_best.pth", map_location=device))
    model.eval()

    with torch.no_grad():
    # 1) 如果不是 Stage 1，就先讀回上一階段的 log
        prev_log = {}
        if stage > 1:
            prev_path = pathlib.Path(save_dir) / f"{reco_code}_stage{stage-1}_log.json"
            if prev_path.exists():
                with open(prev_path, "r") as f:
                    prev_log = json.load(f)

        # 2) 評估 Seg
        miou = eval_seg(model, eval_loaders["seg"], device)
        if stage == 1:
            log["mIoU_base"] = miou
        else:
            # 拿階段 1 的 baseline
            base = prev_log.get("mIoU_base", None)
            log["mIoU_base"]  = base
            log[f"mIoU_stage{stage}"] = miou
            log["mIoU_drop"]  = None if base is None else base - miou

        # 3) 評估 Det
        if stage >= 2:
            mAP = eval_det(model, eval_loaders["det"], device)
            if stage == 2:
                log["mAP_base"] = mAP
            else:
                base = prev_log.get("mAP_base", None)
                log["mAP_base"]     = base
                log[f"mAP_stage{stage}"] = mAP
                log["mAP_drop"]     = None if base is None else base - mAP

        # 4) 評估 Cls (只有 Stage 3)
        if stage == 3:
            top1 = eval_cls(model, eval_loaders["cls"], device)
            log["Top1_base"] = top1

    return model, log