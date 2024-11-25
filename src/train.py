import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from typing import List, Tuple

def compile_model(model, learning_rate, clip_norm):
    """
    Compile the Keras model with Adam optimizer and categorical crossentropy loss function.

    Args:
        model (Model): The Keras model to compile.
        learning_rate (float): Learning rate for the Adam optimizer.
        clip_norm (float): Maximum gradient norm for gradient clipping.

    Returns:
        Model: The compiled Keras model.
    """
    optimizer = Adam(lr=learning_rate, clipnorm=clip_norm)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_callbacks(checkpoint_path, patience):
    """
    Create Keras callbacks for model training.

    Args:
        checkpoint_path (str): File path to save the best model weights.
        patience (int): Number of epochs with no improvement after which training will be stopped.

    Returns:
        list: List of Keras callbacks including ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau.
    """
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience // 2, min_lr=0.001)
    return [checkpoint, early_stopping, reduce_lr]

def train_model(model, X, y, batch_size, epochs, validation_split, callbacks):
    """
    Train the Keras model on the provided dataset.

    Args:
        model (Model): The Keras model to train.
        X (np.ndarray): Input features for training.
        y (np.ndarray): Target labels for training.
        batch_size (int): Number of samples per batch of computation.
        epochs (int): Number of epochs to train the model.
        validation_split (float): Fraction of the training data to be used as validation data.
        callbacks (list): List of Keras callbacks to use during training.

    Returns:
        History: A record of training loss values and metrics values at successive epochs.
    """
    return model.fit([X, y], y, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=callbacks)

def save_artifact(obj, path):
    """
    Save a Python object to a specified file path using pickle.

    Args:
        obj (object): The Python object to save.
        path (str): File path where the object should be saved.
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def plot_training_history(history):
    """
    Plot the training and validation loss and accuracy from the Keras History object.

    Args:
        history (History): Keras History object containing the training and validation loss and accuracy.
    """
    plt.figure(figsize=(12, 6))

    # Plot accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()
    
def load_cnn_daily_mail_dataset(directory: str):
    stories = []
    summaries = []

    # Define the paths to the story and highlight directories
    story_dir = os.path.join(directory, 'stories')
    highlight_dir = os.path.join(directory, 'highlights')

    # List all the story files
    for story_file in os.listdir(story_dir):
        # Read the story file
        with open(os.path.join(story_dir, story_file), 'r', encoding='utf-8') as file:
            story_text = file.read()
        
        # Read the corresponding highlight file
        highlight_file = story_file.replace('.story', '.highlight')
        with open(os.path.join(highlight_dir, highlight_file), 'r', encoding='utf-8') as file:
            highlight_text = file.read()
        
        # Append to lists
        stories.append(story_text)
        summaries.append(highlight_text)
    
    return stories, summaries
