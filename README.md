# Whisper ASR Fine-Tuning Project

This repository contains code for fine-tuning the Whisper model on the ATCO2-ASR dataset using the Hugging Face Transformers library. The goal of this project is to train a speech-to-text (ASR) model for transcribing audio into text, specifically for the ATCO2-ASR dataset.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Requirements](#requirements)
3. [How to Run the Scripts](#how-to-run-the-scripts)
    - [Step 1: Data Preprocessing](#step-1-data-preprocessing)
    - [Step 2: Model Training](#step-2-model-training)
    - [Step 3: Model Evaluation](#step-3-model-evaluation)
4. [Training Configuration](#training-configuration)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Results](#results)
7. [License](#license)

## Project Structure

whisper_asr_finetuning/ │ ├── data_preprocessing.py # Script for loading and preprocessing the dataset ├── model_training.py # Script for defining and training the model ├── evaluation.py # Script for evaluating the model and computing metrics ├── requirements.txt # List of required Python packages ├── outputs/ # Directory for saving trained models and results ├── README.md # Project documentation └── utils/ # Helper scripts (optional, for extended functionality)

python
Copy
Edit

## Requirements

To run the scripts, you'll need to install the required dependencies. Create a Python virtual environment and install the dependencies from the `requirements.txt` file:

### Setting up the environment:

1. **Create a virtual environment** (if you don't have one already):
   ```bash
   python -m venv whisper_env
Activate the virtual environment:

On Windows:

bash
Copy
Edit
.\whisper_env\Scripts\activate
On macOS/Linux:

bash
Copy
Edit
source whisper_env/bin/activate
Install the dependencies:

bash
Copy
Edit
pip install -r requirements.txt
requirements.txt includes:

datasets[audio] — Hugging Face datasets library with audio support

transformers — Hugging Face Transformers library

evaluate — For evaluation metrics

jiwer — Word Error Rate (WER) metric computation

tensorboard — For logging training progress

gradio — For creating interactive demos (optional)

huggingface_hub — To interact with Hugging Face's model hub

How to Run the Scripts
Step 1: Data Preprocessing
This script loads and preprocesses the ATCO2-ASR dataset, converts audio files into features, and tokenizes the text labels for training.

To run the preprocessing script:

bash
Copy
Edit
python data_preprocessing.py
What happens?

The ATCO2-ASR dataset is loaded from Hugging Face.

Audio files are converted into features using WhisperFeatureExtractor.

Text data is tokenized using the WhisperTokenizer.

The preprocessed dataset is saved to be used in the next steps.

Step 2: Model Training
After preprocessing the data, you can proceed with training the model. This script loads the Whisper model, defines the training configuration, and trains the model using the Seq2SeqTrainer from Hugging Face.

To run the training script:

bash
Copy
Edit
python model_training.py
What happens?

The Whisper model and tokenizer are loaded.

The training process begins using the dataset prepared in Step 1.

The model is trained for 10 epochs.

Logs are generated and saved using TensorBoard.

The trained model is saved to the outputs/ directory.

Step 3: Model Evaluation
Once the model is trained, you can evaluate its performance using the Word Error Rate (WER) metric. This script computes WER by comparing the predicted transcriptions to the actual text labels.

To run the evaluation script:

bash
Copy
Edit
python evaluation.py
What happens?

The model saved from Step 2 is loaded.

The WER for both the training and validation sets is computed.

Results are saved to a file called results.txt in the outputs/ directory.

Training Configuration
The training configuration is set in the model_training.py script. The following key parameters are defined:

epochs: Number of training epochs (default: 10)

batch_size: Batch size for training (default: 16)

learning_rate: Learning rate for optimizer (default: 0.00001)

gradient_accumulation_steps: Number of gradient accumulation steps (default: 2)

warmup_steps: Number of warm-up steps for learning rate scheduling (default: 1000)

gradient_checkpointing: Enabled to reduce memory usage during training

eval_strategy: Strategy for evaluation (default: 'epoch')

save_strategy: Strategy for saving model checkpoints (default: 'epoch')

Feel free to modify these parameters in the script to fit your hardware and training needs.

Evaluation Metrics
The primary evaluation metric used is Word Error Rate (WER), which is a common metric for speech-to-text models. It measures the number of word-level transcription errors.

WER = (substitutions + deletions + insertions) / total words

The WER results will be printed to the console and saved to the results.txt file in the outputs/ directory.

Results
After training and evaluation, the results (WER) are saved in outputs/results.txt.

The trained model, tokenizer, and feature extractor are saved in the outputs/ directory as well.

These can be used for inference or further fine-tuning.
