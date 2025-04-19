import argparse
import os
import json
import logging
import torch
import math
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.transforms.v2 as transforms_v2
import torchvision.models as models
from model import VAE
from utils.dataset.nih import NIHChestDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from torchvision import transforms


# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_classifier(args):
    # Argparse
    SAVE_PATH = args.save_path
    IMAGES_PATH = args.images_path
    LABELS_PATH = args.labels_path
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    DEVICE = args.device
    IMAGE_DIM = args.image_dim
    NUM_WORKERS = args.num_workers
    PATHS = [SAVE_PATH]
        
    for path in PATHS:
        if not os.path.exists(path):
            os.makedirs(path)
            
    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()
        
    # Prepare Model
    num_classes = 14
    model = models.resnet50(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Prepare Dataset
    images, labels = np.load(IMAGES_PATH), np.load(LABELS_PATH)
    dataset = NIHChestDataset(images, labels, transform)
    ratio = math.floor(len(dataset) * 0.8)
    
    # Prepare Dataloader
    train_dataloader = DataLoader(dataset[:ratio], BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_dataloader = DataLoader(dataset[ratio:], BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Criterion
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    model.train()
    for epoch in range(NUM_EPOCHS):
        for batch_images, batch_labels in tqdm(train_dataloader, desc=f"Epoch: {epoch + 1}"):
            batch_images, batch_labels = batch_images.to(DEVICE), batch_labels.to(DEVICE)
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # Test Loop
    model.eval()
    with torch.no_grad():
        for batch_images, batch_labels in tqdm(test_dataloader):
            batch_images, batch_labels = batch_images.to(DEVICE), batch_labels.to(DEVICE)
            outputs = model(batch_images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training Script')
    # Data and Save Location
    parser.add_argument('--images_path', type=str, default=os.path.join('datasets', 'nih_custom', 'test_images.npy'))
    parser.add_argument('--labels_path', type=str, default=os.path.join('datasets', 'nih_custom', 'test_labels.npy'))
    parser.add_argument('--save_path', type=str, default=os.path.join('model_weights', 'clean'))

    # Training Configuration
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--image_dim', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    train_classifier(args)