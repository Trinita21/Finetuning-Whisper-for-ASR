Whisper ASR Fine-Tuning
This repository contains code for fine-tuning the Whisper model on the ATCO2-ASR dataset using the Hugging Face Transformers library. The goal is to train a speech-to-text model for transcription tasks.

Requirements
To run the scripts, ensure you have the following Python packages installed:

bash
Copy
Edit
pip install -r requirements.txt
The requirements.txt includes:

datasets[audio]

transformers

evaluate

jiwer

tensorboard

gradio

huggingface_hub

Running the Scripts
1. Data Preprocessing (data_preprocessing.py)
This script loads and preprocesses the ATCO2-ASR dataset, which is required for training the Whisper model.

To run the script:

bash
Copy
Edit
python data_preprocessing.py
It will:

Load the dataset from Hugging Face.

Convert audio to features using the WhisperFeatureExtractor.

Tokenize text and prepare the dataset for training.

2. Model Training (model_training.py)
This script is responsible for defining the model, setting up the training configuration, and training the model on the prepared dataset.

To run the script:

bash
Copy
Edit
python model_training.py
It will:

Load the Whisper model and tokenizer.

Train the model using the Seq2SeqTrainer.

Save the trained model, tokenizer, and processor to the outputs/ directory.

3. Evaluation (evaluation.py)
This script evaluates the trained model using Word Error Rate (WER) as the evaluation metric.

To run the script:

bash
Copy
Edit
python evaluation.py
It will:

Load the best model saved from training.

Compute the WER for both training and validation sets.

4. Results
The results of the evaluation (WER scores) are saved in a results.txt file inside the outputs/ directory.

Training and Evaluation Flow
Step 1: Preprocess the data

Run the data_preprocessing.py script first to prepare the dataset.

Step 2: Train the model

After preprocessing, run the model_training.py script to train the model.

Step 3: Evaluate the model

Once training is complete, evaluate the model by running the evaluation.py script.

Notes
The model is trained for 10 epochs with a batch size of 16. You can adjust these parameters as needed in the model_training.py file.

The training and evaluation logs will be printed to the console and saved to the outputs/ directory.

Ensure you have access to the Hugging Face hub by logging in using huggingface_hub.login() before running the scripts.