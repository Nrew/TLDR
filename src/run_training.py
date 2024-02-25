"""
run_training.py

This script orchestrates the training process of the text summarization model. It includes steps such as data loading,
preprocessing, model compilation, training, and saving the trained model and other artifacts.

The model used is based on an encoder-decoder architecture with attention mechanisms and is trained using the
CNN/Daily Mail dataset for the task of text summarization.

Usage:
    python run_training.py

Requirements:
    - train.py: Contains functions related to model training and evaluation.
    - data_preprocessing.py: Contains functions for data loading and preprocessing.
    - model.py: Contains the model architecture definition.
    - constants.py: Contains configuration constants like file paths and hyperparameters.

Outputs:
    - Trained model weights saved to CHECKPOINT_PATH.
    - Tokenizer object saved to TOKENIZER_PATH.
    - Training history plot displayed and data saved to TRAINING_HISTORY_PATH.
"""

from train import train_model, compile_model, create_callbacks, plot_training_history, save_artifact
from data_preprocessing import prepare_data, load_glove_embeddings, load_dataset
from model import build_model
from constants import *

def run_training():
    """
    Orchestrates the training process including data loading, preprocessing, model training, and saving results.
    """
    # Load and preprocess data
    texts, summaries = load_dataset(TRAINING_DATA_PATH)  # Ensure you have a function to load your dataset
    X, y, tokenizer = prepare_data(texts, summaries, MAX_SEQ_LENGTH, VOCAB_SIZE)
    print("Data has been readied\n")
    
    # Load GloVe embeddings
    embedding_matrix = load_glove_embeddings(GLOVE_PATH, EMBEDDING_DIM, tokenizer)
    print("GloVe has been initalized\n")

    # Build and compile the model
    model = build_model(MAX_SEQ_LENGTH, VOCAB_SIZE)
    embedding_layer = model.get_layer('embedding')
    embedding_layer.set_weights([embedding_matrix])
    embedding_layer.trainable = False
    model = compile_model(model, LEARNING_RATE, clip_norm=1.0)
    print("Model - built and compiled!\n")

    # Create callbacks and train the model
    callbacks = create_callbacks(CHECKPOINT_PATH, patience=10)
    history = train_model(model, X, y, BATCH_SIZE, EPOCHS, validation_split=0.2, callbacks=callbacks)

    # Save artifacts and plot training history
    save_artifact(tokenizer, TOKENIZER_PATH)
    save_artifact(history.history, TRAINING_HISTORY_PATH)
    plot_training_history(history)

if __name__ == '__main__':
    run_training()
