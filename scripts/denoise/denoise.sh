#!/bin/bash

python ./mc_dropout/vae/test.py
python ./mc_dropout/vae_with_custom_loss/test.py
python ./mc_dropout/vae_with_skip_connections/test.py
python ./mc_dropout/vae_with_skip_connections_and_custom_loss/test.py