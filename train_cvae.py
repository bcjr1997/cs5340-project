import argparse
import os
import json
import logging
import torch
import pandas as pd
import torchvision.transforms.v2 as transforms_v2
from models.conditional_vae import CVAE
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
    TRAIN_DATASET_PATH = args.train_dataset_path
    DEV_DATASET_PATH = args.dev_dataset_path
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
    model = CVAE(IMAGE_DIM).to(DEVICE)
    
    noisy_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
        transforms.ToTensor(),
        transforms_v2.GaussianNoise(),
        transforms.Normalize(0.5, 0.5),
    ])
    
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    # Prepare Dataset
    train_df = pd.read_json(TRAIN_DATASET_PATH)
    train_dataset = NIHChestDataset(train_df, transform, noisy_transform)
    dev_df = pd.read_json(DEV_DATASET_PATH)
    dev_dataset = NIHChestDataset(dev_df, transform, noisy_transform)

    # Prepare Dataloader
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    dev_dataloader = DataLoader(dev_dataset, BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Metrics
    psnr = PeakSignalNoiseRatio()
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    # Train Model
    for epoch in range(NUM_EPOCHS):
        train_total_loss = 0
        train_progress_bar = tqdm(train_dataloader)
        model.train()
        for noisy_images, clean_images, labels in train_progress_bar:
            noisy_images, clean_images, labels = noisy_images.to(DEVICE), clean_images.to(DEVICE), labels.to(DEVICE)
            denoised_images, mean, log_variance = model(noisy_images, labels)
            loss = model.loss_function(denoised_images, clean_images, mean, log_variance) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total_loss += loss.item()
            train_progress_bar.set_description(f"Epoch: {epoch + 1} / {NUM_EPOCHS}. Loss: {train_total_loss/len(train_dataloader):.4f}")
        
        dev_total_loss = 0
        dev_progress_bar = tqdm(dev_dataloader)
        model.eval()
        with torch.no_grad():
            for noisy_images, clean_images, labels in dev_progress_bar:
                noisy_images, clean_images, labels = noisy_images.to(DEVICE), clean_images.to(DEVICE), labels.to(DEVICE)
                denoised_images, mean, log_variance = model(noisy_images, labels)
                loss = model.loss_function(denoised_images, clean_images, mean, log_variance) 
                dev_total_loss += loss.item()
                dev_progress_bar.set_description(f"Epoch: {epoch + 1} / {NUM_EPOCHS}. Loss: {dev_total_loss/len(dev_dataloader):.4f}")

        torch.save(model.state_dict(), os.path.join(MODEL_WEIGHTS_PATH, f"vae_weights_{epoch + 1}.pth"))
        logging.info("Model Saved")
    logging.info("Training complete!")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training Script')
    # Data and Save Location
    parser.add_argument('--train_dataset_path', type=str, default=os.path.join('datasets', 'nih_custom', 'train_dataset.json'))
    parser.add_argument('--dev_dataset_path', type=str, default=os.path.join('datasets', 'nih_custom', 'dev_dataset.json'))
    parser.add_argument('--save_path', type=str, default=os.path.join('model_outputs', 'cvae'))

    # Training Configuration
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--image_dim', type=int, default=224)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    train_vae(args)