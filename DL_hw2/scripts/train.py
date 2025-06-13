import os
import torch
from torch.optim import LBFGS
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from utils import get_loaders, train_, yolo_v1_loss
from models.DLHW2Net_v2 import DLHW2Net
import pathlib
import json
## init_lr 設定為1，據說是LBFGS對lr不敏感 ##

def main(init_lr   = 1e-4,
         n_epoch   = 40,
         batch_size= 32,
         n_workers = 5,
         data_root = pathlib.Path(__file__).resolve().parent.parent / "data",
         save_dir  = pathlib.Path(__file__).resolve().parent / "models" / "trained",
         reco_code = ""):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Use {device}")

    # 1) 準備 model (從 ImageNet checkpoint warm-up 開始)
    model = DLHW2Net().to(device)

    # 2) 讀取三個任務的 loader
    loaders = get_loaders('seg',batch_size,n_workers,data_root,device)
    seg_train, seg_val = loaders["train"], loaders['val']

    loaders = get_loaders('det',batch_size,n_workers,data_root,device)
    det_train, det_val = loaders["train"], loaders['val']

    loaders = get_loaders('cls',batch_size,n_workers,data_root,device)
    cls_train, cls_val = loaders["train"], loaders['val']

    eval_loaders = {"seg": seg_val, "det": det_val, "cls": cls_val}

    # 3) 儲存每個 stage 的 log
    all_logs = {}

    # 4) 依序跑 Stage 1 → 2 → 3
    for stage in (1, 2, 3):
        print(f"\n=== Start Stage {stage} ===")

        # 4.1) 重設 optimizer & scheduler（或依需求接續之前的 optimizer 狀態）
        '''
        optimizer = LBFGS(model.parameters(),
                          lr=init_lr,
                          line_search_fn="strong_wolfe")
        '''
        optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=1e-8)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=1,
            eta_min=1e-7,
            last_epoch=-1
        )

        loss_func = {
            1: torch.nn.CrossEntropyLoss(ignore_index=255),
            2: yolo_v1_loss,
            # 2: torch.nn.BCEWithLogitsLoss(),
            3: torch.nn.CrossEntropyLoss()
        }[stage]

        if stage == 1:
            train_loader, val_loader = seg_train, seg_val
        elif stage == 2:
            train_loader, val_loader = det_train, det_val
        else:
            train_loader, val_loader = cls_train, cls_val

        # 4.4) 呼叫 train_，回傳訓練好的 model 以及 log
        model, log = train_(
            model=model,
            stage=stage,
            train_loader=train_loader,
            val_loader=val_loader,
            eval_loaders=eval_loaders,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            save_dir = save_dir,
            reco_code = reco_code,
            init_lr=init_lr,
            n_epoch=n_epoch
        )

        all_logs[f"stage{stage}"] = log

        # 4.5) 存檔
        if save_dir is None:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            save_dir = os.path.join(script_dir, "models", "trained")
        os.makedirs(save_dir, exist_ok=True)

        with open(save_dir / f"{reco_code}_stage{stage}_log.json", "w") as fp:
            json.dump(log, fp, indent=2)

    print("finished")

if __name__ == '__main__':
    import argparse, time

    p = argparse.ArgumentParser()
    default_code = time.strftime("%Y%m%d%H%M")
    p.add_argument("--init_lr", type=float, default=1e-4)
    p.add_argument("--n_epoch", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_workers", type=int, default=5)
    p.add_argument("--reco_code", type=str, default=default_code)
    args = p.parse_args()
    main( init_lr=args.init_lr, 
         n_epoch=args.n_epoch, 
         batch_size=args.batch_size, 
         n_workers=args.n_workers,
         reco_code=args.reco_code)