import argparse
import os
import json
import logging
import torch
import math
import torch.nn as nn
import numpy as np
import pandas as pd
import torchvision.models as models
from utils.dataset.nih import NIHChestDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from torchvision.transforms.v2 import RandomHorizontalFlip, ToTensor, Compose
from torch.utils.data import random_split


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
    num_classes = 12
    model = models.resnet50(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(model.fc.in_features, num_classes)
    )
    model = model.to(DEVICE)
    
    # Transform
    transform = Compose([
        RandomHorizontalFlip(p=0.5),
        ToTensor()
    ])
    
    # Prepare Dataset
    images, labels = np.load(IMAGES_PATH), np.load(LABELS_PATH)
    dataset = NIHChestDataset(images, labels, transform)
    ratio = math.floor(len(dataset) * 0.8)
    train_set, test_set = random_split(dataset, [ratio, len(dataset) - ratio])
    
    # Prepare Dataloader
    train_dataloader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_dataloader = DataLoader(test_set, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Criterion
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    model.train()
    history = {
        'epoch': [],
        'accuracy': [],
        'loss': []
    }
    for epoch in range(NUM_EPOCHS):
        correct, dataset_length, total_loss = 0, 0, 0
        progress_bar = tqdm(train_dataloader)
        for batch_images, batch_labels in progress_bar:
            batch_images, batch_labels = batch_images.to(DEVICE), batch_labels.to(DEVICE)
            
            # Convert one-hot to class indices
            trues = torch.argmax(batch_labels, dim=1).long()
            
            outputs = model(batch_images)
            loss = criterion(outputs, trues)
            
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == trues).sum().item()
            dataset_length += trues.size(0)
            total_loss += loss.item()
            
            progress_bar.set_description(f"Train | Epoch: {epoch + 1} | Accuracy: {(correct / dataset_length) * 100} | Loss: {total_loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        history['epoch'].append(epoch + 1)
        history['accuracy'].append((correct / dataset_length) * 100)
        history['loss'].append(total_loss)
        
    df = pd.DataFrame(history)
    df.to_json(os.path.join(SAVE_PATH, 'train.json'))
            
    # Test Loop
    model.eval()
    history = {
        'accuracy': [],
        'loss': []
    }
    with torch.no_grad():
        progress_bar = tqdm(test_dataloader)
        correct, dataset_length, total_loss = 0, 0, 0
        for batch_images, batch_labels in progress_bar:
            batch_images, batch_labels = batch_images.to(DEVICE), batch_labels.to(DEVICE)
            
            # Convert one-hot to class indices
            trues = torch.argmax(batch_labels, dim=1).long()
            
            outputs = model(batch_images)
            loss = criterion(outputs, trues)
            
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == trues).sum().item()
            dataset_length += trues.size(0)
            total_loss += loss.item()
            
            progress_bar.set_description(f"Test | Accuracy: {(correct / dataset_length) * 100} | Loss: {total_loss}")
            
        history['accuracy'].append((correct / dataset_length) * 100)
        history['loss'].append(total_loss)
            
    df = pd.DataFrame(history)
    df.to_json(os.path.join(SAVE_PATH, 'test.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training Script')
    # Data and Save Location
    parser.add_argument('--images_path', type=str, default=os.path.join('datasets', 'nih_custom', 'test_images.npy'))
    parser.add_argument('--labels_path', type=str, default=os.path.join('datasets', 'nih_custom', 'test_labels.npy'))
    parser.add_argument('--save_path', type=str, default=os.path.join('classification', 'model_outputs', 'clean'))

    # Training Configuration
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    train_classifier(args)