from torch.utils.data import Dataset
from utils.constants.labels import PYTORCH_LABELS
import torch
import torch.nn.functional as F

class NIHChestDataset(Dataset):
    def __init__(self, images, labels, transform, noisy_transform):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.noisy_transform = noisy_transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # Load image on demand
        img= self.images[index]

        # Apply transformations dynamically
        noisy_img = self.noisy_transform(img)
        clean_img = self.transform(img)

        # Convert label to one-hot efficiently
        # label = F.one_hot(torch.tensor(self.labels[index]), num_classes=len(PYTORCH_LABELS)).long()
        label = F.one_hot(
            torch.tensor(self.labels[index], dtype=torch.long),  # <--- Add dtype=torch.long
            num_classes=len(PYTORCH_LABELS)
        )
                
        return noisy_img, clean_img, label
