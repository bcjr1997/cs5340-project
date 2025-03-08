from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import torch.nn.functional as F

LABELS = {
    'Atelectasis': 0,
    'Cardiomegaly': 1,
    'Effusion': 2,
    'Inflitration': 3,
    'Mass': 4,
    'Nodule': 5,
    'Pneumonia': 6,
    'Pneumothorax': 7,
    'Consolidation': 8,
    'Edema': 9,
    'Emphysema': 10,
    'Fibrosis': 11
}

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
        label = F.one_hot(torch.tensor(self.labels[index]), num_classes=len(LABELS)).long()
        
        return noisy_img, clean_img, label
