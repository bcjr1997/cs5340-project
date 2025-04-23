import argparse
import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import pandas as pd
from uuid import uuid4
import torchvision.transforms.v2 as transforms_v2
from model import VAE
from utils.dataset.nih import NIHChestDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def visualize_uncertainties(input_img, recon_img, epistemic,
                            aleatoric, total, save_path="uncertainty_combined.png", to_save=True):
    if to_save == True:
        # Create figure and custom layout
        fig = plt.figure(figsize=(20, 5))
        gs = gridspec.GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 1, 2.5])  # Last panel wider

        # Image displays
        axs = [plt.subplot(gs[i]) for i in range(5)]

        axs[0].imshow(input_img.squeeze().cpu(), cmap='gray')
        axs[0].set_title("Input Image")
        axs[0].axis('off')

        axs[1].imshow(recon_img.squeeze().cpu(), cmap='gray')
        axs[1].set_title("Mean Prediction")
        axs[1].axis('off')

        axs[2].imshow(epistemic.squeeze().cpu(), cmap='hot')
        axs[2].set_title("Epistemic Uncertainty")
        axs[2].axis('off')

        axs[3].imshow(aleatoric.squeeze().cpu(), cmap='hot')
        axs[3].set_title("Aleatoric Uncertainty")
        axs[3].axis('off')

        axs[4].imshow(total.squeeze().cpu(), cmap='hot')
        axs[4].set_title("Total Uncertainty")
        axs[4].axis('off')

        # Flatten uncertainties for line plot
        e_flat = epistemic.squeeze().cpu().numpy().flatten()
        a_flat = aleatoric.squeeze().cpu().numpy().flatten()
        t_flat = total.squeeze().cpu().numpy().flatten()

        # Sort all based on total uncertainty
        sort_idx = np.argsort(t_flat)
        e_sorted = e_flat[sort_idx]
        a_sorted = a_flat[sort_idx]
        t_sorted = t_flat[sort_idx]

        x = np.linspace(0, 1, len(e_flat))

        ax_line = plt.subplot(gs[5])
        ax_line.plot(x, e_sorted, label="Epistemic", linestyle='-', linewidth=1.2, color='royalblue')
        ax_line.plot(x, a_sorted, label="Aleatoric", linestyle='--', linewidth=1.2, color='orange')
        ax_line.plot(x, t_sorted, label="Total", linestyle='-.', linewidth=1.5, color='green')
        ax_line.set_title("Uncertainty (Flattened)")
        ax_line.set_xlabel("Normalized Pixel Index")
        ax_line.set_ylabel("Uncertainty")
        ax_line.grid(True, alpha=0.3)
        ax_line.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=500)
        plt.close()
    else:
        # Flatten uncertainties for line plot
        e_flat = epistemic.squeeze().cpu().numpy().flatten()
        a_flat = aleatoric.squeeze().cpu().numpy().flatten()
        t_flat = total.squeeze().cpu().numpy().flatten()

        # Sort all based on total uncertainty
        sort_idx = np.argsort(t_flat)
        e_sorted = e_flat[sort_idx]
        a_sorted = a_flat[sort_idx]
        t_sorted = t_flat[sort_idx]
    
    return e_sorted, a_sorted, t_sorted

def deep_ensemble_test(args):
    # Argparse
    RANDOM_SEEDS = args.random_seeds
    NOISE_STD = args.noise_std
    noise_stds = [float(x) for x in NOISE_STD.split(',')]
    random_seeds = [int(x) for x in RANDOM_SEEDS.split(',')]
    
    for noise_std in noise_stds:
        SAVE_PATH = args.save_path
        WEIGHTS_PATH = args.weights_path
        SAVE_PATH = f"{SAVE_PATH}_noise_std_{noise_std}"
        TEST_IMAGES_PATH = args.test_images_path
        TEST_LABELS_PATH = args.test_labels_path
        MODEL_WEIGHTS_PATHS = [os.path.join(f"{WEIGHTS_PATH}_random_seed_{seed}_noise_std_{noise_std}", 'model_weights', 'vae_weights_30.pth') for seed in random_seeds]
        BATCH_SIZE = args.batch_size
        DEVICE = args.device
        IMAGE_DIM = args.image_dim
        NUM_WORKERS = args.num_workers
        
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
            
        # Save current args to output
        with open(f"{SAVE_PATH}/argsparse_config.json", 'w') as file:
            json.dump(vars(args), file)
            file.close()

        # Prepare Models
        ensemble_models = []
        for model_weight_path in MODEL_WEIGHTS_PATHS:
            model = VAE(IMAGE_DIM).to(DEVICE)
            model.load_state_dict(torch.load(model_weight_path, weights_only=True))
            model.eval()
            ensemble_models.append(model)
        
        
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
        test_dataloader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True)
        
        # Progress Bar
        test_progress_bar = tqdm(test_dataloader)

        # Eval Model
        history = {
            'epistemic': [],
            'aleatoric': [],
            'total': []
        }
        for noisy_images, clean_images, labels in test_progress_bar:
            noisy_images, clean_images = noisy_images.to(DEVICE), clean_images.to(DEVICE)
            labels = torch.argmax(labels, dim = 1).cpu().numpy()
                
            model_means, model_vars = [], []
            for model in ensemble_models:
                model.eval()
                with torch.no_grad():
                    model_mean, model_aleatoric = model.samples_for_UQ(noisy_images)
                    model_means.append(model_mean)
                    model_vars.append(model_aleatoric)
                
            mean_stack = torch.stack(model_means)  # [N, B, C, H, W]
            var_stack = torch.stack(model_vars)    # [N, B, C, H, W]

            final_prediction = mean_stack.mean(dim=0)
            epistemic_uncertainty = mean_stack.var(dim=0)
            aleatoric_uncertainty = var_stack.mean(dim=0)
            total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
            
            num_samples_to_visualize = min(5, noisy_images.shape[0])
            random_indices = np.random.choice(noisy_images.shape[0], num_samples_to_visualize, replace=False)

            for idx in range(noisy_images.shape[0]):
                filename = f"{uuid4().hex}.jpg"
                image_save_path = os.path.join(SAVE_PATH, 'figures', f"{labels[idx]}")
                if not os.path.exists(image_save_path):
                    os.makedirs(image_save_path)
                e_sorted, a_sorted, t_sorted = visualize_uncertainties(
                    input_img=noisy_images[idx],
                    recon_img=final_prediction[idx],
                    epistemic=epistemic_uncertainty[idx],
                    aleatoric=aleatoric_uncertainty[idx],
                    total=total_uncertainty[idx],
                    save_path=os.path.join(image_save_path, filename),
                    to_save=idx in random_indices
                )
                history['epistemic'].append(e_sorted.mean().item())
                history['aleatoric'].append(a_sorted.mean().item())
                history['total'].append(t_sorted.mean().item())
        
        logging.info('Saving Results')
        result_df = pd.DataFrame(history)
        result_df.to_json(os.path.join(SAVE_PATH, 'results.json'))
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Test Script')
    # Data and Save Location
    parser.add_argument('--test_images_path', type=str, default=os.path.join('datasets', 'nih_custom', 'test_images.npy'))
    parser.add_argument('--test_labels_path', type=str, default=os.path.join('datasets', 'nih_custom', 'test_labels.npy'))
    parser.add_argument('--weights_path', type=str, default=os.path.join('deep_ensembles', 'model_outputs', 'vae_with_skip_connections_and_custom_loss'))
    parser.add_argument('--save_path', type=str, default=os.path.join('deep_ensembles', 'ensemble_model_outputs', 'vae_with_skip_connections_and_custom_loss'))

    # Inference Configuration
    parser.add_argument('--random_seeds', type=str, default="0,42,1337,2024,9999")
    parser.add_argument('--noise_std', type=str, default="0.1,0.5,0.7")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--image_dim', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    deep_ensemble_test(args)