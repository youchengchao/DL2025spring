import os
import pathlib          # build directory
import urllib.request   # download coco through url
import zipfile          # extract compressed files
from tqdm import tqdm   # progress bar
from pycocotools.coco import COCO
import numpy as np
from PIL import Image   # read images
import matplotlib.pyplot as plt  # print images

import random           # sample images
import shutil
import json   # write annotations for mini-COCO

def download(url: str, dst: pathlib.Path, chunk: int = 1 << 20):
    if dst.exists():
        print(f'{dst.name} 已存在，跳過')
        return
    with urllib.request.urlopen(url) as resp, open(dst, 'wb') as fh:
        total = int(resp.headers['Content-Length'])
        bar = tqdm(total=total, unit='B', unit_scale=True, desc=dst.name)
        while True:
            block = resp.read(chunk)
            if not block:
                break
            fh.write(block)
            bar.update(len(block))
        bar.close()

def download_COCO2017(urls: dict):
    base_dir = pathlib.Path(os.path.join(os.getcwd(), 'coco2017'))
    base_dir.mkdir(parents=True, exist_ok=True)

    for url in urls.values():
        fname = url.split('/')[-1]
        download(url, base_dir / fname)

    for zfile in base_dir.glob('*.zip'):
        print(f'Unzipping {zfile.name} ...')
        with zipfile.ZipFile(zfile) as zf:
            zf.extractall(base_dir)

def show_tree(path: pathlib.Path, prefix: str = "", line_limit: int = 30, _counter=[0]):
    """
    簡單印出樹狀結構，最多列印 line_limit 行
    """
    if _counter[0] >= line_limit:
        return
    print(f"{prefix}{path.name}")
    _counter[0] += 1
    if path.is_dir():
        for item in sorted(path.iterdir()):
            if _counter[0] < line_limit:
                show_tree(item, prefix + "    ", line_limit, _counter)
            else:
                break

def watch_coco(data_dir: str, caption_file=None, instance_file=None, keypoint_file=None, num_images=1, spec_json=False):
    """
    data_dir: 圖片目錄（train2017 或 val2017）
    三個 annotation 檔擇一傳入，其餘留 None
    """
    ann_file = caption_file if caption_file else instance_file if instance_file else keypoint_file
    if ann_file is None:
        raise ValueError("必須傳入 caption_file、instance_file 或 keypoint_file 其中之一")
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()

    for i in range(min(num_images, len(img_ids))):
        if spec_json:
            print(coco.loadImgs(img_ids[i])[0])
        img_info = coco.loadImgs(img_ids[i])[0]
        img_path = os.path.join(data_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        plt.imshow(img)
        plt.axis('off')
        ann_ids = coco.getAnnIds(imgIds=img_ids[i])
        anns = coco.loadAnns(ann_ids)
        coco.showAnns(anns)
        plt.show()

def sample_images(anns: dict, selected_cat_ids: list[int], num_per_cat: int, random_state=None, verbose=False) -> dict:
    """
    anns: COCO annotation dict (train_anns 或 val_anns)
    selected_cat_ids: 要抽樣的 category list
    num_per_cat: 每個 category 希望抽到的圖片數量
    回傳格式: {cat_id1: [img_id1, img_id2, ...], ...}
    """
    if random_state is None:
        random_state = random.randint(0, 1000000)
    random.seed(random_state)

    # 先建立 category -> 圖片 ID 的對應
    cat_img_map = {cat_id: set() for cat_id in selected_cat_ids}
    for ann in anns['annotations']:
        cid = ann['category_id']
        if cid in selected_cat_ids:
            cat_img_map[cid].add(ann['image_id'])

    sampled = {}
    used_img_ids = set()  # 已被選過的圖片

    id2name = {cat['id']: cat['name'] for cat in train_anns['categories']}
    pbar = tqdm(selected_cat_ids)
    for cat_id in pbar:
        cat_name = id2name.get(cat_id, str(cat_id))
        pbar.set_description(f"正在抽 {cat_name}")
        
        available = list(cat_img_map[cat_id])
        if len(available) == 0:
            if verbose:
                print(f"[警告] 類別 {cat_id} 根本沒有任何圖片。")
            sampled[cat_id] = []
            continue

        if len(available) < num_per_cat:
            if verbose:
                print(f"[警告] 類別 {cat_id} 可用圖片數 {len(available)} 少於 {num_per_cat}，會全部選取。")
            chosen = available.copy()
        else:
            # 一直重抽，直到抽到跟已用集合無交集
            trial_state = random_state
            while True:
                chosen = random.sample(available, num_per_cat)
                if used_img_ids.isdisjoint(chosen):
                    if verbose:
                        print(f"[Info] 類別 {cat_id} 使用種子 {trial_state} 成功選取。")
                    break
                trial_state += 1
                random.seed(trial_state)
            # （若一直抽不到也會繼續嘗試）
        sampled[cat_id] = chosen
        used_img_ids.update(chosen)

    return sampled

def get_IDs(sample_dict: dict[int, list[int]], random_state=None) -> set[int]:
    """
    從 sample_images 回傳的 dict (cat_id -> [img_id,...])，合併成一個 set
    """
    if random_state is not None:
        random.seed(random_state)
    sid = set()
    for lst in sample_dict.values():
        sid.update(lst)
    return sid

def get_imgNames(anns: dict, ids: set[int]) -> list[str]:
    """
    anns['images'] 裡篩出 id 在 ids 裡面的 file_name
    """
    # 為了效率，先把 id->filename 建 dict
    id2name = {img['id']: img['file_name'] for img in anns['images']}
    return [id2name[i] for i in ids if i in id2name]

def get_imgInfo(anns: dict, ids: set[int]) -> list[dict]:
    """
    anns['images'] 裡篩出 id 在 ids 裡面的整筆 dict
    """
    return [img for img in anns['images'] if img['id'] in ids]

def get_anns(anns: dict, imgIDs: set[int], catIDs: list[int]) -> list[dict]:
    """
    anns['annotations'] 裡篩出 image_id 在 imgIDs 且 category_id 在 catIDs 的所有 annotation dict
    """
    return [ann for ann in anns['annotations'] if (ann['image_id'] in imgIDs and ann['category_id'] in catIDs)]

def copy_images(img_names: list[str], src_dir: str | pathlib.Path, dst_dir: str | pathlib.Path):
    """
    將 src_dir 底下名為 img_names 的檔案，複製到 dst_dir
    """
    for fname in tqdm(img_names, desc=f"Copying to {dst_dir.name}"):
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        if not os.path.exists(dst_path):
            shutil.copy(src_path, dst_path)

if __name__ == "__main__":
    # __file__ : project/scripts/build_miniCOCO.py
    # data directory: project/data
    script_dir = pathlib.Path(__file__).resolve().parent        # .../project/scripts
    project_dir = script_dir.parent                             # .../project
    root_dir = project_dir / "data"                             # .../project/data
    os.chdir(root_dir)

    # （若已下載 COCO2017，可把下面三行註解）
    '''
    urls = {
        'train_images': 'http://images.cocodataset.org/zips/train2017.zip',      # 19 GB
        'val_images'  : 'http://images.cocodataset.org/zips/val2017.zip',        # 1 GB
        'annotations' : 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'  # 770 MB
    }
    download_COCO2017(urls)
    '''

    # 印出 data 資料夾結構
    show_tree(root_dir, line_limit=20)

    # 範例：watch 兩張 captions 的圖片
    watch_coco(
        data_dir=str(root_dir / "coco2017" / "val2017"),
        caption_file=str(root_dir / "coco2017" / "annotations_trainval2017" / "annotations" / "captions_val2017.json"),
        num_images=2,
        spec_json=True
    )

    watch_coco(
        data_dir=str(root_dir / "coco2017" / "val2017"),
        caption_file=str(root_dir / "coco2017" / "annotations_trainval2017" / "annotations" / "instances_val2017.json"),
        num_images=2,
        spec_json=True
    )

    watch_coco(
        data_dir=str(root_dir / "coco2017" / "val2017"),
        caption_file=str(root_dir / "coco2017" / "annotations_trainval2017" / "annotations" / "person_keypoints_val2017.json"),
        num_images=2,
        spec_json=True
    )

    # --------------- 以下開始製作 mini-COCO ---------------
    coco_dir = root_dir / "coco2017"
    mini_coco_dir = root_dir / "mini_coco_det"

    mini_coco_dir.mkdir(parents=True, exist_ok=True)
    (mini_coco_dir / "train").mkdir(parents=True, exist_ok=True)
    (mini_coco_dir / "val").mkdir(parents=True, exist_ok=True)
    (mini_coco_dir / "annotations").mkdir(parents=True, exist_ok=True)

    # 載入完整的 train/val instances
    train_ann_path = coco_dir / "annotations_trainval2017" / "annotations" / "instances_train2017.json"
    val_ann_path   = coco_dir / "annotations_trainval2017" / "annotations" / "instances_val2017.json"
    with open(train_ann_path, 'r') as f:
        train_anns = json.load(f)
    with open(val_ann_path, 'r') as f:
        val_anns = json.load(f)

    # 隨機選 10 個 category (固定 seed)
    all_cat_ids = [cat['id'] for cat in train_anns['categories']]
    random.seed(20250605)
    selected_cat_ids = random.sample(all_cat_ids, 10)

    id_map = {orig: idx for idx, orig in enumerate(selected_cat_ids)}  #### 把挑出來的id重新映射到0~9

    # 每個 category 分別從 train/val 抽圖：train 每類 24 張、val 每類 6 張
    train_samples = sample_images(train_anns, selected_cat_ids, num_per_cat=24, random_state=20250605, verbose=False)
    val_samples   = sample_images(val_anns, selected_cat_ids, num_per_cat=6, random_state=20250605, verbose=False)

    mini_train_ids = get_IDs(train_samples)
    mini_val_ids   = get_IDs(val_samples)

    # 取得對應的檔名 & JSON info
    mini_train_fnames = get_imgNames(train_anns, mini_train_ids)
    mini_val_fnames   = get_imgNames(val_anns, mini_val_ids)

    mini_train_info = get_imgInfo(train_anns, mini_train_ids)
    mini_val_info   = get_imgInfo(val_anns, mini_val_ids)

    mini_train_anns = get_anns(train_anns, mini_train_ids, selected_cat_ids)
    mini_val_anns   = get_anns(val_anns, mini_val_ids, selected_cat_ids)

    for ann in mini_train_anns:
        ann['category_id'] = id_map[ann['category_id']]
    for ann in mini_val_anns:
        ann['category_id'] = id_map[ann['category_id']]
    new_categories = [
        {'id': id_map[cat['id']], 'name': cat['name']}
        for cat in train_anns['categories']
        if cat['id'] in selected_cat_ids
    ]

    # 寫出新的 instances_train.json、instances_val.json
    with open(mini_coco_dir / "annotations" / "instances_train.json", 'w') as f:
        json.dump({
            'images': mini_train_info,
            'annotations': mini_train_anns,
            'categories': new_categories
        }, f)

    with open(mini_coco_dir / "annotations" / "instances_val.json", 'w') as f:
        json.dump({
            'images': mini_val_info,
            'annotations': mini_val_anns,
            'categories': new_categories
        }, f)

    # 最後把圖片複製到 mini_coco_det/train、mini_coco_det/val
    copy_images(mini_train_fnames, root_dir / "coco2017" / "train2017", mini_coco_dir / "train")
    copy_images(mini_val_fnames,   root_dir / "coco2017" / "val2017",   mini_coco_dir / "val")
