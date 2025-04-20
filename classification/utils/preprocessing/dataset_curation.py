import argparse
import os
import json
import logging
import numpy as np
import cv2
from tqdm import tqdm
# from utils.constants.labels import RAW_LABELS

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_dataset(args):
    # Paths
    SAVE_PATH = args.save_path
    DATASET_PATH = args.dataset_path
    
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # Save current args to output
    with open(os.path.join(SAVE_PATH, "argsparse_config.json"), 'w') as file:
        json.dump(vars(args), file)
    
    # List all files
    for model in os.listdir(DATASET_PATH):
        model_path = os.path.join(DATASET_PATH, model)
        for noise_level in os.listdir(model_path):
            noise_path = os.path.join(model_path, noise_level)
            denoise_image_path = os.path.join(noise_path, 'parsed_images', 'denoised')
            images, labels = [], []
            for label in os.listdir(denoise_image_path):
                label_path = os.path.join(denoise_image_path, label)
                for image_name in tqdm(os.listdir(label_path), desc=f"Model: {model} | Noise Level: {noise_level} | Label: {label}"):
                    img_path = os.path.join(label_path, image_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Ensure grayscale loading
                    images.append(img)
                    labels.append(label)
            curr_save_path = os.path.join(SAVE_PATH, model, noise_level)
            os.makedirs(curr_save_path, exist_ok=True)
            denoised_images = np.array(images)
            denoised_labels = np.array(labels)
            np.save(os.path.join(curr_save_path, 'denoised_images.npy'), denoised_images)
            np.save(os.path.join(curr_save_path, 'denoised_labels.npy'), denoised_labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Preprocessing Script')
    parser.add_argument('--dataset_path', type=str, default=os.path.join('denoised_outputs'))
    parser.add_argument('--save_path', type=str, default=os.path.join('datasets', 'denoised_nih_custom'))
    args = parser.parse_args()
    
    preprocess_dataset(args)