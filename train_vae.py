import argparse
import os
import json
import logging
from models.vae import VAE
from utils.dataset.nih import NIHChestDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_vae(args):
    # Argparse
    SAVE_PATH = args.save_path
    DATASET_PATH = args.dataset_path
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    DEVICE = args.device
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()

    # Prepare Model
    model = VAE()

    # Prepare Dataset
    train_dataset = NIHChestDataset()

    # Prepare Dataloader
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)

    # Optimizer
    optimizer = Adam(model.parameters, lr=LEARNING_RATE)

    # Train Model
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for noisy_images, clean_images in tqdm(train_dataloader, desc=f"Epoch: {epoch + 1} / {NUM_EPOCHS}"):
            noisy_images, clean_images = noisy_images.to(DEVICE), clean_images.to(DEVICE)
            denoised_images, mean, log_variance = model(noisy_images)
            loss = model.loss_function(denoised_images, clean_images, mean, log_variance) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss/len(train_dataloader):.4f}")
    
    logging.info("Training complete!")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training Script')
    # Data and Save Location
    parser.add_argument('--dataset_path', type=str, default=os.path.join('datasets'))
    parser.add_argument('--save_path', type=str, default=os.path.join('model_outputs', 'vae'))

    # Training Configuration
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--image_dim', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    train_vae(args)