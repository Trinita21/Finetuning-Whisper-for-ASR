#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Run the data preprocessing script
python data_preprocessing.py

# Run the model training script
python model_training.py

# Evaluate the model
python evaluation.py
