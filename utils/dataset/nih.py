from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

class NIHChestDataset(Dataset):
    def __init__(self, dataframe, transform, noisy_transform):
        self.dataframe = dataframe
        self.images = self.dataframe['image_path'].tolist()
        self.labels = self.dataframe['label'].tolist()
        self.classes = list(set(self.labels))
        self.transform = transform
        self.noisy_transform = noisy_transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]
        img = Image.open(img_path)
        noisy_img = self.noisy_transform(img)
        clean_img = self.transform(img)
        label = torch.Tensor([1 if label == x else 0 for x in self.classes]).long()
        label = F.one_hot(label, num_classes=len(self.classes))
        return noisy_img, clean_img, label