
import os
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path

url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"

if __name__ == "__main__":
    random.seed(20250606)
    # 1. 定位 script_dir、project_dir、root_dir
    script_dir  = Path(__file__).resolve().parent       # .../project/scripts
    project_dir = script_dir.parent                      # .../project
    root_dir    = project_dir / "data"                   # .../project/data
    root_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(root_dir)

    # 2. 下載 .tgz 檔到 root_dir（如果已存在就跳過）
    tgz_path = root_dir / "imagenette2.tgz"
    if not tgz_path.exists():
        print(f"Downloading {url} → {tgz_path.name}")
        urllib.request.urlretrieve(url, tgz_path)
    else:
        print(f"{tgz_path.name} 已存在，跳過下載")

    # 3. 解壓到 root_dir
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=root_dir)
    # 通常解壓後會在 data/ 下產生一個名為 "imagenette2" 的資料夾
    imagenette_dir = root_dir / "imagenette2"
    imagenette160_dir = root_dir / "imagenette160"

    # 4. 準備輸出資料夾結構
    train_out = imagenette160_dir / "train"
    val_out   = imagenette160_dir / "val"
    for d in (train_out, val_out):
        d.mkdir(parents=True, exist_ok=True)

    # 5. 取得所有 class（通常 imagenette2/train 底下的子目錄即為類別）
    train_dir = imagenette_dir / "train"
    val_dir = imagenette_dir / "val"
    classes = [p.name for p in train_dir.iterdir() if p.is_dir()]
    # 如果超過 10 類，就隨機挑 10 類
    if len(classes) > 10:
        classes = random.sample(classes, k=10)
    print("Selected classes:", classes)

    # 6. 對每個 class 進行抽樣並複製
    for cls in classes:
        # 6.1 建立該 class 在輸出資料夾的子目錄
        (train_out / cls).mkdir(parents=True, exist_ok=True)
        (val_out   / cls).mkdir(parents=True, exist_ok=True)

        # 6.2 Train：從 imagenette2/train/<cls> 底下隨機抽 24 張
        src_train_cls = train_dir / cls
        all_train_imgs = list(src_train_cls.glob("*"))
        sampled_train = random.sample(all_train_imgs, k=min(24, len(all_train_imgs)))
        for img_path in sampled_train:
            shutil.copy(img_path, train_out / cls / img_path.name)

        # 6.3 Val：從 imagenette2/train/<cls> 底下隨機抽 24 張
        src_val_dir = val_dir / cls
        all_val_imgs = list(src_val_dir.glob("*"))
        sampled_val = random.sample(all_val_imgs, k=min(6, len(all_val_imgs)))
        for img_path in sampled_val:
            shutil.copy(img_path, val_out / cls / img_path.name)

    print(f"已完成抽樣，結果存於：{imagenette160_dir}")
