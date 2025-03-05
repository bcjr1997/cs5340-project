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
        'image_paths': [],
        'labels': []
    }
    for folder in os.listdir(DATASET_PATH):
        if 'images_' in folder:
            curr_path = os.path.join(DATASET_PATH, folder, 'images')
            images = os.listdir(curr_path)
            dataset['image_paths'].extend([os.path.join(curr_path, img_name) for img_name in images])
            label_idx = int(re.search('\d+', folder).group())
            dataset['labels'].extend([LABELS[label_idx] for _ in range(len(images))])

    df = pd.DataFrame(dataset)
    df.to_json(os.path.join(SAVE_PATH, 'parsed_dataset.json'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training Script')
    # Data and Save Location
    parser.add_argument('--dataset_path', type=str, default=os.path.join('datasets', 'nih'))
    parser.add_argument('--save_path', type=str, default=os.path.join('datasets', 'nih_custom'))
    args = parser.parse_args()
    preprocess_dataset(args)