#!/bin/bash

# Deep Ensembles
python ./deep_ensembles/vae/deep_ensemble_test.py                                                                                                                                
python ./deep_ensembles/vae_with_custom_loss/deep_ensemble_test.py
python ./deep_ensembles/vae_with_skip_connections/deep_ensemble_test.py
python ./deep_ensembles/vae_with_skip_connections_and_custom_loss/deep_ensemble_test.py