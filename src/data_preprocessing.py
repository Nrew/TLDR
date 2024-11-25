import os

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from typing import List, Tuple
from constants import CHUNK_THRESHOLD

def chunk_text(text: str, chunk_size: int) -> List[str]:
    """
    Splits a text into smaller chunks of a specified size.

    Args:
      text (str): The text to be chunked.
      chunk_size (int): The number of words in each chunk.

    Returns:
      chunks (List[str]): List of text chunks.
    """
    
    words = text.split(' ')
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def prepare_data(texts: List[str], summaries: List[str], max_seq_length: int, vocab_size: int, chunk_threshold: int = CHUNK_THRESHOLD) -> Tuple[List[np.ndarray], List[np.ndarray], Tokenizer]:
    """
    Preprocesses input texts and summaries for a natural language processing task.

    Args:
      texts (List[str]): List of input texts.
      summaries (List[str]): List of corresponding summaries.
      max_seq_length (int): Maximum length for padding sequences.
      vocab_size (int): Size of the vocabulary for tokenization.
      chunk_threshold (int): Maximum number of words in a text before it is chunked.


    Returns:
      X_chunks (List[np.ndarray]): Padded and tokenized input sequences.
      y_chunks (List[np.ndarray]): Padded and tokenized target sequences (summaries).
      tokenizer (Tokenizer): Trained tokenizer for converting texts to sequences.
    """
    
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts + summaries)
    
    X_chunks, y_chunks = [], []
    
    for text, summary in zip(texts, summaries):
        if len(text.split()) > chunk_threshold:
            # If text is longer than the threshold, chunk it
            text_chunks = chunk_text(text, chunk_threshold)
            summary_chunks = chunk_text(summary, chunk_threshold)
            chunks = zip(text_chunks, summary_chunks)
        else:
            # If text is shorter than the threshold, process it as is
            chunks = [(text, summary)]
        
        for text_chunk, summary_chunk in chunks:
            # Convert chunks to sequences
            input_sequences = tokenizer.texts_to_sequences([text_chunk])
            target_sequences = tokenizer.texts_to_sequences([summary_chunk])
            
            # Pad sequences to a fixed length
            X = pad_sequences(input_sequences, maxlen=max_seq_length, padding='post', truncating='post')
            y = pad_sequences(target_sequences, maxlen=max_seq_length, padding='post', truncating='post')
            
            X_chunks.append(X)
            y_chunks.append(y)
    
    return X_chunks, y_chunks, tokenizer

def load_glove_embeddings(path: str, embedding_dim: int, tokenizer: Tokenizer) -> np.ndarray:
    """
    Load GloVe word embeddings from a specified file path.

    Args:
        path (str): Path to the GloVe embeddings file.
        embedding_dim (int): Dimensionality of the GloVe word vectors.
        tokenizer (Tokenizer): Tokenizer object used for the text data.

    Returns:
        embedding_matrix (np.ndarray): A matrix of shape (vocab_size, embedding_dim) containing the GloVe word vectors.
    """
    
    embeddings_index = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def load_dataset(directory: str) -> Tuple[List[str], List[str]]:
    """
    Loads the stories and highlights from the CNN/Daily Mail dataset.

    Args:
    - directory (str): The directory where the CNN/Daily Mail dataset is located.

    Returns:
    - texts (List[str]): The list of stories.
    - summaries (List[str]): The list of corresponding highlights (summaries).
    """
    texts = []
    summaries = []
    
    # List all the story files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Check if the file is a story file
        if os.path.isfile(file_path) and filename.endswith('.story'):
            with open(file_path, 'r', encoding='utf-8') as file:
                story_file = file.read()
                
                # Split the story and highlights
                story_parts = story_file.split('@highlight')
                story_text = story_parts[0].strip()
                story_highlights = [highlight.strip() for highlight in story_parts[1:]]
                
                # Concatenate highlights with a period
                story_summary = '. '.join(story_highlights)
                
                # Append the story and concatenated highlights to their respective lists
                texts.append(story_text)
                summaries.append(story_summary)
    
    return texts, summaries

def load_dataset(directory: str) -> Tuple[List[str], List[str]]:
    """
    Loads the stories and highlights from the CNN/Daily Mail dataset, traversing all subdirectories.

    Args:
      directory (str): The root directory where the CNN/Daily Mail dataset is located.

    Returns:
      texts (List[str]): The list of stories.
      summaries (List[str]): The list of corresponding highlights (summaries).
    """
    texts = []
    summaries = []
    
    # Walk through all files and subdirectories in the given directory
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.story'):
                file_path = os.path.join(dirpath, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    story_file = file.read()
                    
                    # Split the story and highlights
                    story_parts = story_file.split('@highlight')
                    story_text = story_parts[0].strip()
                    story_highlights = [highlight.strip() for highlight in story_parts[1:]]
                    
                    # Concatenate highlights with a space or a period (depending on your preference)
                    story_summary = ' '.join(story_highlights)
                    
                    # Append the story and concatenated highlights to their respective lists
                    texts.append(story_text)
                    summaries.append(story_summary)
    
    return texts, summaries