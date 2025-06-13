import matplotlib.pyplot as plt
from utils import get_loaders
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    for stage in ["seg", "det", "cls"]:
        loaders = get_loaders(stage)
        for name, dataloader in loaders.items():
            match stage:
                case "seg": name = "miniVOC_"+name
                case "det": name = "miniCOCO_"+name
                case "cls": name = "imagenette160_"+name
            print(f"Showing samples from: {name}")
            batch = next(iter(dataloader))
            # 處理三種不同的 dataset 格式
            if name.startswith("imagenette160"):
                images, labels = batch  # classification
            elif name.startswith("miniCOCO"):
                images, targets = batch  # detection
            elif name.startswith("miniVOC"):
                images, masks = batch  # segmentation
            else:
                continue

            # 顯示前兩張圖像
            fig, axs = plt.subplots(1, 2, figsize=(6, 3))
            for i in range(2):
                img = images[i]
                img = img.permute(1, 2, 0).numpy()
                img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # unnormalize
                img = img.clip(0, 1)
                axs[i].imshow(img)
                axs[i].set_title(f"{name} sample {i}")
                axs[i].axis('off')
            plt.tight_layout()
            plt.show()