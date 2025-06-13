import os
import urllib.request as request 
import pathlib
import shutil
import random
import tarfile

from PIL import Image
import numpy as np

# PASCAL VOC 2012 : http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
'''
Annotation(.XML):
<annotation>
    <filename><filename>
    <folder><folder>
    <source><source>
    <size><size>
    <object><object>
<annotation>
'''

# VOC 2012 20 classes 與其對應的 label ID（SegmentationClass PNG 中的像素值）
CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
# 對應的 ID 剛好是 index+1
CLASS_TO_ID = {name: idx + 1 for idx, name in enumerate(CLASS_NAMES)}

def download_and_extract_voc2012(dest_dir):
    url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    tar_path = os.path.join(dest_dir, "VOCtrainval_11-May-2012.tar")
    if not os.path.exists(os.path.join(dest_dir, "VOCdevkit", "VOC2012")):
        print("Downloading Pascal VOC 2012...")
        request.urlretrieve(url, tar_path)
        print("Extracting...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=dest_dir)
        os.remove(tar_path) # 刪除 .tar 檔
        # tar_path.unlink() # 刪除 .tar 檔
        print("Download and extraction complete.")
    else:
        print("VOC2012 dataset already exists.")

def build_mini_voc_seg(root_dir: pathlib.Path, mini_dir_name="mini_voc_seg",
                       train_per_class=12, val_per_class=3, seed=42):
    random.seed(seed)
    voc_root = root_dir / "VOCtrainval_11-May-2012" / "VOCdevkit" / "VOC2012"
    if not voc_root.exists():
        raise FileNotFoundError(f"找不到 {voc_root}，請先下載並解壓 VOC2012。")

    orig_img_dir = voc_root / "JPEGImages"
    orig_seg_dir = voc_root / "SegmentationClass"

    # mini_voc_seg 目錄結構
    mini_root = root_dir / mini_dir_name
    mini_img_dir = mini_root / "JPEGImages"
    mini_seg_dir = mini_root / "SegmentationClass"
    mini_imageset_dir = mini_root / "ImageSets" / "Segmentation"

    for p in (mini_img_dir, mini_seg_dir, mini_imageset_dir):
        p.mkdir(parents=True, exist_ok=True)

    # 先讀取所有 SegmentationClass 檔案名稱 (不含副檔名)
    all_mask_paths = list(orig_seg_dir.glob("*.png"))
    all_ids = [p.stem for p in all_mask_paths]

    # 複製用的 train_ids, val_ids 集合
    train_ids = set()
    val_ids = set()

    # 針對每個 class 進行抽樣
    for cls_name in CLASS_NAMES:
        cls_id = CLASS_TO_ID[cls_name]
        # 找出所有含此 class 的影像 ID
        imgs_with_cls = []
        for img_id in all_ids:
            mask_path = orig_seg_dir / f"{img_id}.png"
            mask = np.array(Image.open(mask_path))
            if (mask == cls_id).any():
                imgs_with_cls.append(img_id)

        # 若該類影像數量不夠，則全部取用
        n_train = min(train_per_class, len(imgs_with_cls))
        n_val = min(val_per_class, len(imgs_with_cls) - n_train)
        sampled = random.sample(imgs_with_cls, k=min(len(imgs_with_cls), train_per_class + val_per_class))
        sampled_train = sampled[:n_train]
        sampled_val = sampled[n_train : n_train + n_val]

        print(f"Class '{cls_name}' (ID={cls_id}) 共 {len(imgs_with_cls)} 張影像，抽取 {len(sampled_train)} train / {len(sampled_val)} val")

        # 複製到 mini 目錄，並加入 ID 到集合裡
        for img_id in sampled_train:
            # 影像檔 複製
            src_img = orig_img_dir / f"{img_id}.jpg"
            dst_img = mini_img_dir / f"{img_id}.jpg"
            shutil.copy(src_img, dst_img)
            # segmentation mask 複製
            src_mask = orig_seg_dir / f"{img_id}.png"
            dst_mask = mini_seg_dir / f"{img_id}.png"
            shutil.copy(src_mask, dst_mask)
            train_ids.add(img_id)

        for img_id in sampled_val:
            src_img = orig_img_dir / f"{img_id}.jpg"
            dst_img = mini_img_dir / f"{img_id}.jpg"
            shutil.copy(src_img, dst_img)
            src_mask = orig_seg_dir / f"{img_id}.png"
            dst_mask = mini_seg_dir / f"{img_id}.png"
            shutil.copy(src_mask, dst_mask)
            val_ids.add(img_id)

    # 寫入 ImageSets/Segmentation/train.txt 與 val.txt
    train_txt = mini_imageset_dir / "train.txt"
    val_txt = mini_imageset_dir / "val.txt"
    with open(train_txt, "w") as f:
        for img_id in sorted(train_ids):
            f.write(f"{img_id}\n")
    with open(val_txt, "w") as f:
        for img_id in sorted(val_ids):
            f.write(f"{img_id}\n")

    print(f"\nmini VOC Segmentation Dataset 已建立於：{mini_root}")
    print(f"共 {len(train_ids)} 張 train, {len(val_ids)} 張 val")
    print("結構如下：")
    print(f"{mini_root}/")
    print("  JPEGImages/ - .jpg 檔")
    print("  SegmentationClass/ - .png 分割遮罩檔")
    print("  ImageSets/Segmentation/train.txt")
    print("  ImageSets/Segmentation/val.txt")

if __name__ == '__main__':
    # __file__ : project/scripts/build_miniCOCO.py
    # data directory: project/data
    script_dir = pathlib.Path(__file__).resolve().parent        # .../project/scripts
    project_dir = script_dir.parent                             # .../project
    root_dir = project_dir / "data"                             # .../project/data
    os.chdir(root_dir)
    # download_and_extract_voc2012(root_dir)
    build_mini_voc_seg(root_dir)