#!/bin/bash

# Monte Carlo Dropout
python ./mc_dropout/vae/mc_dropout_test.py                                                                                                                                
python ./mc_dropout/vae_with_custom_loss/mc_dropout_test.py
python ./mc_dropout/vae_with_skip_connections/mc_dropout_test.py
python ./mc_dropout/vae_with_skip_connections_and_custom_loss/mc_dropout_test.py
