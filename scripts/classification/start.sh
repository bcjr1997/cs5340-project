#!/bin/bash

# Original
python ./classification/eval.py --images_path="./datasets/nih_custom/test_images.npy" --labels_path="./datasets/nih_custom/test_labels.npy" --save_path="./classification/model_outputs/original"

# VAE
python ./classification/eval.py --images_path="./datasets/denoised_nih_custom/vae/0.1/denoised_images.npy" --labels_path="./datasets/denoised_nih_custom/vae/0.1/denoised_labels.npy" --save_path="./classification/model_outputs/vae/0.1"
python ./classification/eval.py --images_path="./datasets/denoised_nih_custom/vae/0.5/denoised_images.npy" --labels_path="./datasets/denoised_nih_custom/vae/0.5/denoised_labels.npy" --save_path="./classification/model_outputs/vae/0.5"
python ./classification/eval.py --images_path="./datasets/denoised_nih_custom/vae/0.7/denoised_images.npy" --labels_path="./datasets/denoised_nih_custom/vae/0.7/denoised_labels.npy" --save_path="./classification/model_outputs/vae/0.7"

# VAE with Custom Loss
python ./classification/eval.py --images_path="./datasets/denoised_nih_custom/vae_with_custom_loss/0.1/denoised_images.npy" --labels_path="./datasets/denoised_nih_custom/vae_with_custom_loss/0.1/denoised_labels.npy" --save_path="./classification/model_outputs/vae_with_custom_loss/0.1"
python ./classification/eval.py --images_path="./datasets/denoised_nih_custom/vae_with_custom_loss/0.5/denoised_images.npy" --labels_path="./datasets/denoised_nih_custom/vae_with_custom_loss/0.5/denoised_labels.npy" --save_path="./classification/model_outputs/vae_with_custom_loss/0.5"
python ./classification/eval.py --images_path="./datasets/denoised_nih_custom/vae_with_custom_loss/0.7/denoised_images.npy" --labels_path="./datasets/denoised_nih_custom/vae_with_custom_loss/0.7/denoised_labels.npy" --save_path="./classification/model_outputs/vae_with_custom_loss/0.7"

# VAE with Skip Connections
python ./classification/eval.py --images_path="./datasets/denoised_nih_custom/vae_with_skip_connections/0.1/denoised_images.npy" --labels_path="./datasets/denoised_nih_custom/vae_with_skip_connections/0.1/denoised_labels.npy" --save_path="./classification/model_outputs/vae_with_skip_connections/0.1"
python ./classification/eval.py --images_path="./datasets/denoised_nih_custom/vae_with_skip_connections/0.5/denoised_images.npy" --labels_path="./datasets/denoised_nih_custom/vae_with_skip_connections/0.5/denoised_labels.npy" --save_path="./classification/model_outputs/vae_with_skip_connections/0.5"
python ./classification/eval.py --images_path="./datasets/denoised_nih_custom/vae_with_skip_connections/0.7/denoised_images.npy" --labels_path="./datasets/denoised_nih_custom/vae_with_skip_connections/0.7/denoised_labels.npy" --save_path="./classification/model_outputs/vae_with_skip_connections/0.7"

# VAE with Skip Connections and Custom Loss
python ./classification/eval.py --images_path="./datasets/denoised_nih_custom/vae_with_skip_connections_and_custom_loss/0.1/denoised_images.npy" --labels_path="./datasets/denoised_nih_custom/vae_with_skip_connections_and_custom_loss/0.1/denoised_labels.npy" --save_path="./classification/model_outputs/vae_with_skip_connections_and_custom_loss/0.1"
python ./classification/eval.py --images_path="./datasets/denoised_nih_custom/vae_with_skip_connections_and_custom_loss/0.5/denoised_images.npy" --labels_path="./datasets/denoised_nih_custom/vae_with_skip_connections_and_custom_loss/0.5/denoised_labels.npy" --save_path="./classification/model_outputs/vae_with_skip_connections_and_custom_loss/0.5"
python ./classification/eval.py --images_path="./datasets/denoised_nih_custom/vae_with_skip_connections_and_custom_loss/0.7/denoised_images.npy" --labels_path="./datasets/denoised_nih_custom/vae_with_skip_connections_and_custom_loss/0.7/denoised_labels.npy" --save_path="./classification/model_outputs/vae_with_skip_connections_and_custom_loss/0.7"