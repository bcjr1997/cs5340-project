import argparse
import os
import json
import logging
import torch
import numpy as np
import pandas as pd
import torchvision.transforms.v2 as transforms_v2
from models.vae import VAE
from utils.dataset.nih import NIHChestDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from torchvision import transforms
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_vae(args):
    # Argparse
    SAVE_PATH = args.save_path
    TRAIN_IMAGES_PATH = args.train_images_path
    TRAIN_LABELS_PATH = args.train_labels_path
    DEV_IMAGES_PATH = args.dev_images_path
    DEV_LABELS_PATH = args.dev_labels_path
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    DEVICE = args.device
    IMAGE_DIM = args.image_dim
    MODEL_WEIGHTS_PATH = os.path.join(SAVE_PATH, 'model_weights')
    PATHS = [SAVE_PATH, MODEL_WEIGHTS_PATH]
    
    for path in PATHS:
        if not os.path.exists(path):
            os.makedirs(path)
        
    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()

    # Prepare Model
    model = VAE(IMAGE_DIM).to(DEVICE)
    
    noisy_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms_v2.GaussianNoise(),
        transforms.Normalize(0.5, 0.5),
    ])
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    # Prepare Dataset
    train_images, train_labels = np.load(TRAIN_IMAGES_PATH), np.load(TRAIN_LABELS_PATH)
    dev_images, dev_labels = np.load(DEV_IMAGES_PATH), np.load(DEV_LABELS_PATH)
    train_dataset = NIHChestDataset(train_images, train_labels, transform, noisy_transform)
    dev_images, dev_labels = np.load(DEV_IMAGES_PATH), np.load(DEV_LABELS_PATH)
    dev_dataset = NIHChestDataset(dev_images, dev_labels, transform, noisy_transform)

    # Prepare Dataloader
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    dev_dataloader = DataLoader(dev_dataset, BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Metrics
    psnr = PeakSignalNoiseRatio().to(DEVICE)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    # Train Model
    for epoch in range(NUM_EPOCHS):
        train_total_loss = 0
        train_progress_bar = tqdm(train_dataloader)
        model.train()
        for noisy_images, clean_images, _ in train_progress_bar:
            noisy_images, clean_images = noisy_images.to(DEVICE), clean_images.to(DEVICE)
            denoised_images, mean, log_variance = model(noisy_images)
            loss = model.loss_function(denoised_images, clean_images, mean, log_variance) 
            psnr_value = psnr(denoised_images, clean_images).item()
            ssim_value = ssim(denoised_images, clean_images).item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total_loss += loss.item()
            train_progress_bar.set_description(f"Epoch: {epoch + 1} / {NUM_EPOCHS}. Loss: {train_total_loss/len(train_dataloader):.4f}, PSNR: {psnr_value:.4f} SSIM: {ssim_value:.4f}")
        
        model.eval()
        dev_total_loss = 0
        dev_progress_bar = tqdm(dev_dataloader)
        with torch.no_grad():
            for noisy_images, clean_images, _ in dev_progress_bar:
                noisy_images, clean_images = noisy_images.to(DEVICE), clean_images.to(DEVICE)
                denoised_images, mean, log_variance = model(noisy_images)
                loss = model.loss_function(denoised_images, clean_images, mean, log_variance) 
                psnr_value = psnr(denoised_images, clean_images).item()
                ssim_value = ssim(denoised_images, clean_images).item()
                dev_total_loss += loss.item()
                dev_progress_bar.set_description(f"Epoch: {epoch + 1} / {NUM_EPOCHS}. Loss: {dev_total_loss/len(dev_dataloader):.4f}, PSNR: {psnr_value:.4f} SSIM: {ssim_value:.4f}")

        torch.save(model.state_dict(), os.path.join(MODEL_WEIGHTS_PATH, f"vae_weights_{epoch + 1}.pth"))
        logging.info("Model Saved")
    logging.info("Training complete!")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training Script')
    # Data and Save Location
    parser.add_argument('--train_images_path', type=str, default=os.path.join('datasets', 'nih_custom', 'train_images.npy'))
    parser.add_argument('--dev_images_path', type=str, default=os.path.join('datasets', 'nih_custom', 'dev_images.npy'))
    parser.add_argument('--train_labels_path', type=str, default=os.path.join('datasets', 'nih_custom', 'train_labels.npy'))
    parser.add_argument('--dev_labels_path', type=str, default=os.path.join('datasets', 'nih_custom', 'dev_labels.npy'))
    parser.add_argument('--save_path', type=str, default=os.path.join('model_outputs', 'vae'))

    # Training Configuration
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--image_dim', type=int, default=224)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    train_vae(args)