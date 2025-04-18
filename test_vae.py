import argparse
import os
import json
import logging
import numpy as np
import torch
import pandas as pd
import numpy as np
import torchvision.transforms.v2 as transforms_v2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from models.vae_V4 import VAE
from utils.dataset.nih import NIHChestDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from torchvision import transforms
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# keep dropout layer active at inference by setting them to train mode
def enable_dropout(model):
    """Enable dropout layers during inference."""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()

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
    MC_PASSES = args.mc_passes  # number of MC forward passes
    
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
    batch_results = []
    model.eval()
    # with torch.no_grad():
    #     for noisy_images, clean_images, _ in test_progress_bar:
    #         noisy_images, clean_images = noisy_images.to(DEVICE), clean_images.to(DEVICE)
    #         denoised_images, mean, log_variance = model(noisy_images)
    #         loss = model.loss_function(denoised_images, clean_images, mean, log_variance, BETA_WEIGHTAGE) 
    #         test_total_loss += loss.item()
    #         psnr_value = psnr(denoised_images, clean_images).item()
    #         ssim_value = ssim(denoised_images, clean_images).item()
    #         test_progress_bar.set_description(f"Loss: {test_total_loss/len(test_dataloader):.4f}, PSNR: {psnr_value:.4f} SSIM: {ssim_value:.4f}")
    
    with torch.no_grad():
        for batch_idx, (noisy_images, clean_images, _) in enumerate(test_progress_bar):
            noisy_images, clean_images = noisy_images.to(DEVICE), clean_images.to(DEVICE)
            
            # Perform MC_PASSES forward passes with dropout enabled
            predictions = []
            means = []
            log_vars = []
            for _ in range(MC_PASSES):
                enable_dropout(model)  # Re-enable dropout layers for this pass
                denoised_images, mean, log_variance = model(noisy_images)
                predictions.append(denoised_images.unsqueeze(0))
                means.append(mean.unsqueeze(0))
                log_vars.append(log_variance.unsqueeze(0))
            
            # Aggregate predictions over MC_PASSES (compute the mean prediction)
            predictions = torch.cat(predictions, dim=0)  # shape: [MC_PASSES, batch, channels, H, W]
            mean_prediction = predictions.mean(dim=0)           

            # variance across MC_PASSES
            uncertainty = predictions.var(dim=0)

            # Aggregate mean and logvar across MC passes
            means = torch.cat(means, dim=0)              # [MC_PASSES, B, latent_dim]
            log_vars = torch.cat(log_vars, dim=0)        # [MC_PASSES, B, latent_dim]
            mean_mu = means.mean(dim=0)
            mean_logvar = log_vars.mean(dim=0)
            
            # Compute loss and metrics using the mean prediction
            loss = model.loss_function(mean_prediction, clean_images, mean_mu, mean_logvar, BETA_WEIGHTAGE)
            test_total_loss += loss.item()
            psnr_value = psnr(mean_prediction, clean_images).item()
            ssim_value = ssim(mean_prediction, clean_images).item()

            # Calculate average uncertainty for the batch (mean over all pixels and channels)
            avg_uncertainty = uncertainty.mean().item()
            max_uncertainty = uncertainty.max().item()
            test_progress_bar.set_description(f"Loss: {test_total_loss/len(test_dataloader):.4f}, PSNR: {psnr_value:.4f} SSIM: {ssim_value:.4f} Max Uncertainty: {max_uncertainty:.4f}")

            # save result by batch
            batch_result = {
                "batch_index": batch_idx,
                "loss": loss.item(),
                "psnr": psnr_value,
                "ssim": ssim_value,
                "avg_uncertainty": avg_uncertainty,
                "max_uncertainty": max_uncertainty,
            }
            batch_results.append(batch_result)

            # Optionally, perform visualization for a few random samples
            batch_size = clean_images.shape[0]
            num_samples_to_visualize = min(10, batch_size)
            random_indices = np.random.choice(batch_size, num_samples_to_visualize, replace=False)
            
            for idx in random_indices:
                save_file = os.path.join(SAVE_PATH, f"uncertainty_batch{batch_idx}_sample{idx}.png")
                visualize_uncertainties(
                    clean_img = clean_images[idx],
                    noisy_img = noisy_images[idx],
                    recon_img = mean_prediction[idx],
                    epistemic = uncertainty[idx],
                    save_path = save_file
                )
            
    # After inference, save the batch metrics to a JSON file
    print(batch_results)
    results_file = os.path.join(SAVE_PATH, "batch_metrics.json")
    with open(results_file, 'w') as f:
        json.dump(batch_results, f, indent=4)

    print(f"Saved batch metrics to {results_file}")

            


def visualize_uncertainties(clean_img, noisy_img, recon_img, epistemic, save_path="uncertainty_combined.png"):
    """
    Visualize and save the following results in one figure:
    - Clean (original) image.
    - Noisy image.
    - Reconstructed image from the model.
    - Epistemic uncertainty heat map.
    - Flattened epistemic uncertainty plot.
    
    Parameters:
        clean_img (Tensor): The original clean image.
        noisy_img (Tensor): The image with added noise.
        recon_img (Tensor): The model's reconstructed output.
        epistemic (Tensor): The epistemic uncertainty (variance across MC passes).
        save_path (str): Where to save the final combined image.
    """
    # Create a figure with a custom layout (5 panels)
    fig = plt.figure(figsize=(24, 5))
    gs = gridspec.GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 1.5, 1.5])
    
    # Display Clean Image
    ax0 = plt.subplot(gs[0])
    ax0.imshow(clean_img.squeeze().cpu(), cmap='gray')
    ax0.set_title("Clean Image")
    ax0.axis('off')
    
    # Display Noisy Image
    ax1 = plt.subplot(gs[1])
    ax1.imshow(noisy_img.squeeze().cpu(), cmap='gray')
    ax1.set_title("Noisy Image")
    ax1.axis('off')
    
    # Display Reconstructed Image
    ax2 = plt.subplot(gs[2])
    ax2.imshow(recon_img.squeeze().cpu(), cmap='gray')
    ax2.set_title("Reconstructed Image")
    ax2.axis('off')
    
    # Display Epistemic Uncertainty Heat Map
    ax3 = plt.subplot(gs[3])
    ax3.imshow(epistemic.squeeze().cpu(), cmap='hot')
    ax3.set_title("Epistemic Uncertainty")
    ax3.axis('off')
    
    # Flatten the epistemic uncertainty for line plotting
    e_flat = epistemic.squeeze().cpu().numpy().flatten()
    # Optionally, sort the uncertainty values for visualization clarity
    sort_idx = np.argsort(e_flat)
    e_sorted = e_flat[sort_idx]
    x = np.linspace(0, 1, len(e_flat))
    
    # Plot Flattened Uncertainty
    ax_line = plt.subplot(gs[4])
    ax_line.plot(x, e_sorted, label="Epistemic", linestyle='-', linewidth=1.2, color='royalblue')
    ax_line.set_title("Uncertainty (Flattened)")
    ax_line.set_xlabel("Normalized Pixel Index")
    ax_line.set_ylabel("Uncertainty")
    ax_line.grid(True, alpha=0.3)
    ax_line.legend()

    # plot cdf
    # cdf = np.linspace(0, 1, len(e_sorted))
    # ax4 = plt.subplot(gs[4])
    # ax4.plot(e_sorted, cdf, color='green', linewidth=2)
    # ax4.set_title("CDF of Uncertainty")
    # ax4.set_xlabel("Uncertainty")
    # ax4.set_ylabel("Cumulative Fraction")
    # ax4.grid(True, alpha=0.3)

    # plot a histogram of uncertainty distribution
    ax5 = plt.subplot(gs[5])
    ax5.hist(e_flat, bins=50, color='skyblue', log=True)
    ax5.set_title("Histogram of Uncertainty(log scale)")
    ax5.set_xlabel("Uncertainty")
    ax5.set_ylabel("Pixel count(log)")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=500)
    plt.close()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training Script')
    # Data and Save Location
    parser.add_argument('--test_images_path', type=str, default=os.path.join('datasets', 'nih_custom', 'test_images.npy'))
    parser.add_argument('--test_labels_path', type=str, default=os.path.join('datasets', 'nih_custom', 'test_labels.npy'))
    parser.add_argument('--save_path', type=str, default=os.path.join('model_outputs', 'vae', 'test'))

    # Inference Configuration
    parser.add_argument('--model_weights', type=str, default=os.path.join('model_outputs', 'vae', 'model_weights', 'vae_weights_1.pth'))
    parser.add_argument('--beta_weightage', type=float, default=1e-4, help='For scaling KL divergence Loss')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--image_dim', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mc_passes', type=int, default=10, help="Number of Monte Carlo forward passes")
    args = parser.parse_args()
    test_vae(args)