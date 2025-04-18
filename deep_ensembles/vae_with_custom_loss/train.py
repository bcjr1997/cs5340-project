import argparse
import os
import json
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.transforms.v2 as transforms_v2
from model import VAE
from utils.dataset.nih import NIHChestDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from torchvision import transforms
from utils.constants.labels import REVERSED_PYTORCH_LABELS
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def save_images(epoch, noisy, clean, denoised, labels, save_path):
    labels = np.argmax(labels, axis=1)  # Convert one-hot labels to class indices
    unique_labels, indices = np.unique(labels, return_index=True)

    # Ensure correct selection along axis 0
    noisy = np.take(noisy, indices, axis=0)
    clean = np.take(clean, indices, axis=0)
    denoised = np.take(denoised, indices, axis=0)

    fig, axes = plt.subplots(3, len(unique_labels), figsize=(20, 10))
    titles = ["Clean", "Noisy", "Denoised"]

    for i, img_set in enumerate([clean, noisy, denoised]):
        for j, img in enumerate(img_set):
            axes[i, j].imshow(img.squeeze(), cmap='gray')  # Ensure grayscale display
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].set_frame_on(False)

    # Fix iteration over `titles` (use `len(titles)`)
    for i in range(len(titles)):
        axes[i, 0].set_ylabel(titles[i], fontsize=12)

    # Fix iteration over `unique_labels`
    for j in range(len(unique_labels)):
        axes[2, j].set_xlabel(REVERSED_PYTORCH_LABELS[unique_labels[j]], fontsize=12)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)  # Ensure directory exists
    plt.savefig(os.path.join(save_path, f'epoch_{epoch}.png'))
    plt.close()

def train_vae(args):
    NOISE_STD = args.noise_std
    RANDOM_SEEDS = args.random_seeds
    random_seeds = [int(x) for x in RANDOM_SEEDS.split(',')]
    noise_stds = [float(x) for x in NOISE_STD.split(',')]
    
    for seed in random_seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        for noise_std in noise_stds:
            # Argparse
            SAVE_PATH = args.save_path
            SAVE_PATH = f"{SAVE_PATH}_random_seed_{seed}_noise_std_{noise_std}"
            BETA_WEIGHTAGE = args.beta_weightage
            TRAIN_IMAGES_PATH = args.train_images_path
            TRAIN_LABELS_PATH = args.train_labels_path
            DEV_IMAGES_PATH = args.dev_images_path
            DEV_LABELS_PATH = args.dev_labels_path
            BATCH_SIZE = args.batch_size
            NUM_EPOCHS = args.epochs
            LEARNING_RATE = args.learning_rate
            DEVICE = args.device
            IMAGE_DIM = args.image_dim
            NUM_WORKERS = args.num_workers
            MODEL_WEIGHTS_PATH = os.path.join(SAVE_PATH, 'model_weights')
            SAVE_IMAGE_PATH = os.path.join(SAVE_PATH, 'sample_images')
            PATHS = [SAVE_PATH, MODEL_WEIGHTS_PATH, SAVE_IMAGE_PATH]
            
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
                transforms_v2.GaussianNoise(noise_std),
                transforms.Normalize(0, 1)
            ])
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(0, 1)
            ])

            # Prepare Dataset
            train_images, train_labels = np.load(TRAIN_IMAGES_PATH), np.load(TRAIN_LABELS_PATH)
            dev_images, dev_labels = np.load(DEV_IMAGES_PATH), np.load(DEV_LABELS_PATH)
            train_dataset = NIHChestDataset(train_images, train_labels, transform, noisy_transform)
            dev_images, dev_labels = np.load(DEV_IMAGES_PATH), np.load(DEV_LABELS_PATH)
            dev_dataset = NIHChestDataset(dev_images, dev_labels, transform, noisy_transform)

            # Prepare Dataloader
            train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
            dev_dataloader = DataLoader(dev_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

            # Optimizer
            optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
            
            # Metrics
            psnr = PeakSignalNoiseRatio().to(DEVICE)
            ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
            
            # History
            history = {"epoch": [], "train_loss": [], "train_psnr": [], "train_ssim": [], "dev_loss": [], "dev_psnr": [], "dev_ssim": []}

            # Train Model
            for epoch in range(NUM_EPOCHS):
                train_total_loss, train_psnr_total, train_ssim_total = 0, 0, 0
                train_progress_bar = tqdm(train_dataloader)
                model.train()
                for noisy_images, clean_images, _ in train_progress_bar:
                    noisy_images, clean_images = noisy_images.to(DEVICE), clean_images.to(DEVICE)
                    denoised_images, mean, log_variance = model(noisy_images)
                    loss = model.loss_function(denoised_images, clean_images, mean, log_variance, BETA_WEIGHTAGE) 
                    psnr_value = psnr(denoised_images, clean_images).item()
                    ssim_value = ssim(denoised_images, clean_images).item()
                    train_psnr_total += psnr_value
                    train_ssim_total += ssim_value
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    train_total_loss += loss.item()
                    train_progress_bar.set_description(f"Epoch: {epoch + 1} / {NUM_EPOCHS}. Loss: {train_total_loss/len(train_dataloader):.4f}, PSNR: {psnr_value:.4f} SSIM: {ssim_value:.4f}")
                
                model.eval()
                dev_total_loss, dev_psnr_total, dev_ssim_total = 0, 0, 0
                dev_progress_bar = tqdm(dev_dataloader)
                with torch.no_grad():
                    final_noisy_images, final_clean_images, final_denoised_images, final_labels = None, None, None, None
                    for noisy_images, clean_images, labels in dev_progress_bar:
                        noisy_images, clean_images = noisy_images.to(DEVICE), clean_images.to(DEVICE)
                        denoised_images, mean, log_variance = model(noisy_images)
                        loss = model.loss_function(denoised_images, clean_images, mean, log_variance, BETA_WEIGHTAGE) 
                        psnr_value = psnr(denoised_images, clean_images).item()
                        ssim_value = ssim(denoised_images, clean_images).item()
                        dev_psnr_total += psnr_value
                        dev_ssim_total += ssim_value
                        dev_total_loss += loss.item()
                        dev_progress_bar.set_description(f"Epoch: {epoch + 1} / {NUM_EPOCHS}. Loss: {dev_total_loss/len(dev_dataloader):.4f}, PSNR: {psnr_value:.4f} SSIM: {ssim_value:.4f}")
                        
                        if final_noisy_images is None:
                            final_noisy_images = np.array(noisy_images.cpu().detach().numpy())
                            final_clean_images = np.array(clean_images.cpu().detach().numpy())
                            final_denoised_images = np.array(denoised_images.cpu().detach().numpy())
                            final_labels = np.array(labels.cpu().detach().numpy())
                        else:
                            final_noisy_images = np.concatenate((final_noisy_images, noisy_images.cpu().detach().numpy()))
                            final_clean_images = np.concatenate((final_clean_images, clean_images.cpu().detach().numpy()))
                            final_denoised_images = np.concatenate((final_denoised_images, denoised_images.cpu().detach().numpy()))
                            final_labels = np.concatenate((final_labels, labels.cpu().detach().numpy()))

                    save_images(epoch, final_noisy_images, final_clean_images, final_denoised_images, final_labels, SAVE_IMAGE_PATH)
                
                history["epoch"].append(epoch + 1)
                history["train_loss"].append(train_total_loss / len(train_dataloader))
                history["train_psnr"].append(train_psnr_total / len(train_dataloader))
                history["train_ssim"].append(train_ssim_total / len(train_dataloader))
                history["dev_loss"].append(dev_total_loss / len(dev_dataloader))
                history["dev_psnr"].append(dev_psnr_total / len(dev_dataloader))
                history["dev_ssim"].append(dev_ssim_total / len(dev_dataloader))
                
            result_df = pd.DataFrame(history)
            result_df.to_json(os.path.join(SAVE_PATH, 'train_results.json'))
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
    parser.add_argument('--save_path', type=str, default=os.path.join('deep_ensembles', 'model_outputs', 'vae_with_custom_loss'))

    # Training Configuration
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--random_seeds', type=str, default="0,42,1337,2024,9999")
    parser.add_argument('--noise_std', type=str, default="0.1,0.5,0.7")
    parser.add_argument('--beta_weightage', type=float, default=1e-4)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--image_dim', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    train_vae(args)