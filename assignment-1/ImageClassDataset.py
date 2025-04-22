from torch.utils.data import Dataset
import os
import torch
from PIL import Image
class ImageClassDataset(Dataset):
    def __init__(self, names_file, transform = None):
        self.images_dir = os.path.join(os.getcwd(), "images")
        self.transform = transform
        with open(os.path.join(self.images_dir, names_file)) as Name_file:
            lines = [line.replace("\n","").split(' ') for line in Name_file]
        self.x = [os.path.join(self.images_dir, parts[0]) for parts in lines]
        self.y = torch.LongTensor([int(parts[1]) for parts in lines])
    
    def __getitem__(self, index):
        img_path = self.x[index]
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.y[index]

    def __len__(self):
        return len(self.x)