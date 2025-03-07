import argparse
import os
import json
import logging
import re
import pandas as pd

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Constants
LABELS = {
    1: 'Atelectasis',
    2: 'Cardiomegaly',
    3: 'Effusion',
    4: 'Inflitration',
    5: 'Mass',
    6: 'Nodule',
    7: 'Pneumonia',
    8: 'Pneumothorax',
    9: 'Consolidation',
    10: 'Edema',
    11: 'Emphysema',
    12: 'Fibrosis'
}

def preprocess_dataset(args):
    # Argparse
    SAVE_PATH = args.save_path
    DATASET_PATH = args.dataset_path
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    
    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()
        
    # List all files
    dataset = {
        'image_path': [],
        'label': []
    }
    for folder in os.listdir(DATASET_PATH):
        if 'images_' in folder:
            curr_path = os.path.join(DATASET_PATH, folder, 'images')
            images = os.listdir(curr_path)
            dataset['image_path'].extend([os.path.join(curr_path, img_name) for img_name in images])
            label_idx = int(re.search('\d+', folder).group())
            dataset['label'].extend([LABELS[label_idx] for _ in range(len(images))])

    df = pd.DataFrame(dataset)
    min_count = df['label'].value_counts().min()
    df = df.groupby('label').apply(lambda x: x.sample(n=min_count, random_state=42)).reset_index(drop=True)
    train_df = df.sample(frac=0.8, random_state=42)
    dev_test_df = df.drop(train_df.index)
    dev_df = dev_test_df.sample(frac=0.5, random_state=42)
    test_df = dev_test_df.drop(dev_df.index)
    logging.info(f"Dataset Length: {len(train_df)} (Train) | {len(dev_df)} (Dev) | {len(test_df)} (Test)")
    train_df.reset_index(drop=True).to_json(os.path.join(SAVE_PATH, 'train_dataset.json'))
    dev_df.reset_index(drop=True).to_json(os.path.join(SAVE_PATH, 'dev_dataset.json'))
    test_df.reset_index(drop=True).to_json(os.path.join(SAVE_PATH, 'test_dataset.json'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training Script')
    # Data and Save Location
    parser.add_argument('--dataset_path', type=str, default=os.path.join('datasets', 'nih'))
    parser.add_argument('--save_path', type=str, default=os.path.join('datasets', 'nih_custom'))
    args = parser.parse_args()
    preprocess_dataset(args)