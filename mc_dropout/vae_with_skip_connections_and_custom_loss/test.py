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

def save_images(noisy, clean, denoised, labels, save_path):
    # Save the Images 
    for index, clean_image in tqdm(enumerate(clean)):
        label = str(np.argmax(labels[index]))
        
        paths = [
            os.path.join(save_path, 'original', label),
            os.path.join(save_path, 'denoised', label),
            os.path.join(save_path, 'noisy', label)
        ]
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
        
        
        # Save Clean Images
        img = clean_image.squeeze(0)
        plt.imsave(os.path.join(os.path.join(save_path, 'original', label), f"img_{index}.png"), img, cmap='gray')
            
        # Save Denoised Images
        denoised_image = denoised[index]
        img = denoised_image.squeeze(0)
        plt.imsave(os.path.join(os.path.join(save_path, 'denoised', label), f"img_{index}.png"), img, cmap='gray')
            
        # Save Noisy Images
        noisy_image = noisy[index]
        img = noisy_image.squeeze(0)
        plt.imsave(os.path.join(os.path.join(save_path, 'noisy', label), f"img_{index}.png"), img, cmap='gray')


def train_vae(args):
    NOISE_STD = args.noise_std
    noise_stds = [float(x) for x in NOISE_STD.split(',')]
    
    for noise_std in noise_stds:
        # Argparse
        SAVE_PATH = args.save_path
        SAVE_PATH = os.path.join(SAVE_PATH, f"{noise_std}")
        MODEL_WEIGHTS_PATH = args.model_weights_path
        MODEL_WEIGHTS_PATH = f"{MODEL_WEIGHTS_PATH}_noise_std_{noise_std}"
        MODEL_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_PATH, 'model_weights', 'vae_weights_30.pth')
        TEST_IMAGES_PATH = args.test_images_path
        TEST_LABELS_PATH = args.test_labels_path
        BATCH_SIZE = args.batch_size
        DEVICE = args.device
        IMAGE_DIM = args.image_dim
        NUM_WORKERS = args.num_workers
        BETA_WEIGHTAGE = args.beta_weightage
        SAVE_IMAGE_PATH = os.path.join(SAVE_PATH, 'parsed_images')
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
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, weights_only=True))
        model.eval()
        
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
        test_images, test_labels = np.load(TEST_IMAGES_PATH), np.load(TEST_LABELS_PATH)
        test_dataset = NIHChestDataset(test_images, test_labels, transform, noisy_transform)

        # Prepare Dataloader
        test_dataloader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        
        # Metrics
        psnr = PeakSignalNoiseRatio().to(DEVICE)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
        
        # History
        history = {"test_loss": [], "test_psnr": [], "test_ssim": []}

        # Train Model
        model.eval()
        with torch.no_grad():
            test_total_loss, test_psnr_total, test_ssim_total = 0, 0, 0
            test_progress_bar = tqdm(test_dataloader)
            final_noisy_images, final_clean_images, final_denoised_images, final_labels = None, None, None, None
            for noisy_images, clean_images, labels in test_progress_bar:
                noisy_images, clean_images = noisy_images.to(DEVICE), clean_images.to(DEVICE)
                denoised_images, mean, log_variance = model(noisy_images)
                loss = model.loss_function(denoised_images, clean_images, mean, log_variance, BETA_WEIGHTAGE) 
                psnr_value = psnr(denoised_images, clean_images).item()
                ssim_value = ssim(denoised_images, clean_images).item()
                test_psnr_total += psnr_value
                test_ssim_total += ssim_value
                test_total_loss += loss.item()
                test_progress_bar.set_description(f"Loss: {test_total_loss/len(test_dataloader):.4f}, PSNR: {psnr_value:.4f} SSIM: {ssim_value:.4f}")
                    
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

            save_images(final_noisy_images, final_clean_images, final_denoised_images, final_labels, SAVE_IMAGE_PATH)
            
            history["test_loss"].append(test_total_loss / len(test_dataloader))
            history["test_psnr"].append(test_psnr_total / len(test_dataloader))
            history["test_ssim"].append(test_ssim_total / len(test_dataloader))
            
        result_df = pd.DataFrame(history)
        result_df.to_json(os.path.join(SAVE_PATH, 'denoised_results.json'))
        logging.info("Denoised complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training Script')
    # Data and Save Location
    parser.add_argument('--test_images_path', type=str, default=os.path.join('datasets', 'nih_custom', 'test_images.npy'))
    parser.add_argument('--test_labels_path', type=str, default=os.path.join('datasets', 'nih_custom', 'test_labels.npy'))
    parser.add_argument('--model_weights_path', type=str, default=os.path.join('mc_dropout', 'model_outputs', 'vae_with_skip_connections_and_custom_loss'))
    parser.add_argument('--save_path', type=str, default=os.path.join('denoised_outputs', 'vae_with_skip_connections_and_custom_loss'))

    # Training Configuration
    parser.add_argument('--noise_std', type=str, default="0.1,0.5,0.7")
    parser.add_argument('--beta_weightage', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--image_dim', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    train_vae(args)