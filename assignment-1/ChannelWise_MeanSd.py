from ImageClassDataset import ImageClassDataset
from DataloaderWrapper import DataloaderWrapper

def compute_dataset_mean_std(filename, transform):
    dataset = ImageClassDataset(filename, transform=transform)
    loader = DataloaderWrapper(
        dataset=dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=5, 
        pin_memory=True
    ).dataloader
    mean = 0.0
    std = 0.0
    nb_samples = 0

    for data, _ in loader:

        B,C,H,W = data.size()
        data = data.view(B, C, H*W)
        
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += B

    mean /= nb_samples
    std /= nb_samples
    return mean, std