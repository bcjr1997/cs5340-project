import argparse
import os
import json
import logging
import numpy as np
import torch
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

def test_vae(args):
    # Argparse
    SAVE_PATH = args.save_path
    TEST_IMAGES_PATH = args.test_images_path
    TEST_LABELS_PATH = args.test_labels_path
    MODEL_WEIGHTS = args.model_weights
    BATCH_SIZE = args.batch_size
    DEVICE = args.device
    IMAGE_DIM = args.image_dim
    NUM_WORKERS = args.num_workers
    BETA_WEIGHTAGE = args.beta_weightage
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        
    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()

    # Prepare Model
    model = VAE(IMAGE_DIM).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_WEIGHTS, weights_only=True))
    
    noisy_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms_v2.GaussianNoise(),
        transforms.Normalize(0, 1)
    ])
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0, 1)
    ])

    # Prepare Dataset
    #test_df = pd.read_json(TEST_DATASET_PATH)
    test_images, test_labels = np.load(TEST_IMAGES_PATH), np.load(TEST_LABELS_PATH)
    test_dataset = NIHChestDataset(test_images, test_labels, transform, noisy_transform)

    # Prepare Dataloader
    test_dataloader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True)
    
    # Progress Bar
    test_progress_bar = tqdm(test_dataloader)
    
    # Metrics
    psnr = PeakSignalNoiseRatio().to(DEVICE)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    # Eval Model
    test_total_loss = 0
    model.eval()
    with torch.no_grad():
        for noisy_images, clean_images, _ in test_progress_bar:
            noisy_images, clean_images = noisy_images.to(DEVICE), clean_images.to(DEVICE)
            denoised_images, mean, log_variance = model(noisy_images)
            loss = model.loss_function(denoised_images, clean_images, mean, log_variance, BETA_WEIGHTAGE) 
            test_total_loss += loss.item()
            psnr_value = psnr(denoised_images, clean_images).item()
            ssim_value = ssim(denoised_images, clean_images).item()
            test_progress_bar.set_description(f"Loss: {test_total_loss/len(test_dataloader):.4f}, PSNR: {psnr_value:.4f} SSIM: {ssim_value:.4f}")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training Script')
    # Data and Save Location
    parser.add_argument('--test_images_path', type=str, default=os.path.join('datasets', 'nih_custom', 'test_images.npy'))
    parser.add_argument('--test_labels_path', type=str, default=os.path.join('datasets', 'nih_custom', 'test_labels.npy'))
    parser.add_argument('--save_path', type=str, default=os.path.join('model_outputs', 'vae', 'test'))

    # Training Configuration
    parser.add_argument('--model_weights', type=str, default=os.path.join('model_outputs', 'vae', 'model_weights', 'vae_weights_1.pth'))
    parser.add_argument('--beta_weightage', type=float, default=1e-4, help='For scaling KL divergence Loss')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--image_dim', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    test_vae(args)