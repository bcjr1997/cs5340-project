#!/bin/bash

# Monte Carlo Dropout
python ./mc_dropout/vae/train.py                                                                                                                                
python ./mc_dropout/vae_with_custom_loss/train.py
python ./mc_dropout/vae_with_skip_connections/train.py
python ./mc_dropout/vae_with_skip_connections_and_custom_loss/train.py

# Deep Ensembles
python ./deep_ensembles/vae/train.py                                                                                                                                
python ./deep_ensembles/vae_with_custom_loss/train.py
python ./deep_ensembles/vae_with_skip_connections/train.py
python ./deep_ensembles/vae_with_skip_connections_and_custom_loss/train.py