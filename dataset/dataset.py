import numpy as np
from skimage import io
from torch.utils.data import Dataset
from PIL import Image

class AnimalDataset(Dataset):
    def __init__(self, image_path, labels, load_transform=None, train_transform=None):
        self.image_path = image_path
        self.labels = labels
        self.load_transform = load_transform
        self.train_transform = train_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, image_id):
        full_path = self.image_path + "/" + self.labels[image_id][0]
        image = Image.open(full_path).convert('RGB')
        label = self.labels[image_id][1]
        if self.train_transform:
            image = self.train_transform(image)
        if self.load_transform:
            image = self.load_transform(image)

        return image, label