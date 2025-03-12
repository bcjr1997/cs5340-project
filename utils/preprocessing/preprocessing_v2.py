import argparse
import os
import json
import logging
import re
import pandas as pd
import numpy as np
import cv2
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from utils.constants.labels import RAW_LABELS

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
label_mapping = {label: idx for idx, label in enumerate(RAW_LABELS.values())}  # Convert labels to numeric

def process_image(args):
    img_path, label, image_dim = args
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Ensure grayscale loading

    if img is None:
        print(f"Skipping image (failed to load): {img_path}")
        return None

    # Ensure valid dtype
    if img.dtype != np.uint8 or len(img.shape) != 2:  # Grayscale images should have shape (H, W)
        print(f"Skipping image due to unsupported format: {img_path}")
        return None

    img = cv2.resize(img, (image_dim, image_dim))  # Resize
    img = np.expand_dims(img, axis=-1)  # Convert to (1, H, W) for PyTorch
    return img, label_mapping[label]


def preprocess_dataset(args):
    # Paths
    SAVE_PATH = args.save_path
    DATASET_PATH = args.dataset_path
    IMAGE_DIM = args.image_dim
    
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # Save current args to output
    with open(os.path.join(SAVE_PATH, "argsparse_config.json"), 'w') as file:
        json.dump(vars(args), file)
    
    # List all files
    dataset = []
    
    for folder in os.listdir(DATASET_PATH):
        if 'images_' in folder:
            curr_path = os.path.join(DATASET_PATH, folder, 'images')
            images = os.listdir(curr_path)
            label_idx = int(re.search(r'\d+', folder).group())
            for img_name in images:
                img_path = os.path.join(curr_path, img_name)
                dataset.append((img_path, RAW_LABELS[label_idx], IMAGE_DIM))
    
    logger.info(f"Processing {len(dataset)} images using {cpu_count()} processes...")
    
    # Process images in parallel with progress bar
    with Pool(processes=cpu_count()) as pool:
        processed_data = list(tqdm(pool.imap(process_image, dataset), total=len(dataset), desc="Processing Images"))
        pool.close()
    
    images, labels = zip(*[data for data in processed_data if data[0] is not None])
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int64)
    
    # Convert to DataFrame for splitting
    df = pd.DataFrame({'images': list(images), 'labels': labels})
    min_count = df['labels'].value_counts().min()
    df = df.groupby('labels').apply(lambda x: x.sample(n=min_count, random_state=42)).reset_index(drop=True)
    
    # Split dataset
    train_df = df.sample(frac=0.8, random_state=42)
    dev_test_df = df.drop(train_df.index)
    dev_df = dev_test_df.sample(frac=0.5, random_state=42)
    test_df = dev_test_df.drop(dev_df.index)
    
    train_images = np.array(train_df['images'].tolist())
    train_labels = np.array(train_df['labels'].tolist())
    dev_images = np.array(dev_df['images'].tolist())
    dev_labels = np.array(dev_df['labels'].tolist())
    test_images = np.array(test_df['images'].tolist())
    test_labels = np.array(test_df['labels'].tolist())
    
    # Save dataset splits
    np.save(os.path.join(SAVE_PATH, 'train_images.npy'), train_images)
    np.save(os.path.join(SAVE_PATH, 'train_labels.npy'), train_labels)
    np.save(os.path.join(SAVE_PATH, 'dev_images.npy'), dev_images)
    np.save(os.path.join(SAVE_PATH, 'dev_labels.npy'), dev_labels)
    np.save(os.path.join(SAVE_PATH, 'test_images.npy'), test_images)
    np.save(os.path.join(SAVE_PATH, 'test_labels.npy'), test_labels)
    
    logger.info(f"Dataset saved with {len(train_images)} (Train), {len(dev_images)} (Dev), {len(test_images)} (Test)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Preprocessing Script')
    parser.add_argument('--dataset_path', type=str, default=os.path.join('datasets', 'nih'))
    parser.add_argument('--save_path', type=str, default=os.path.join('datasets', 'nih_custom'))
    parser.add_argument('--image_dim', type=int, default=224)
    args = parser.parse_args()
    
    preprocess_dataset(args)