"""
generate.py

This script is responsible for generating summaries from new texts using the trained text summarization model.
It includes functionalities for loading the trained model and tokenizer, preprocessing new input texts, 
generating summaries, and post-processing the summaries to ensure coherence and conciseness.

The script also includes an enhanced summary aggregation function that reduces redundancy and maintains the logical flow
of the text, making it suitable for summarizing multiple related chunks of text.

Usage:
    python generate.py

Requirements:
    - Keras for loading the trained model.
    - NumPy for numerical operations.
    - NLTK for text processing.
    - Sklearn for cosine similarity calculation.

Outputs:
    - Generated summaries for the input texts.
"""

import pickle
import numpy as np
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple

from constants import TOKENIZER_PATH, MODEL_PATH, MAX_SEQ_LENGTH


def load_artifacts(tokenizer_path: str, model_path: str) -> Tuple[Tokenizer, Model]:
    """
    Load the tokenizer and trained model from specified file paths.
    
    Args:
        tokenizer_path (str): Path to the saved tokenizer.
        model_path (str): Path to the saved trained model.
        
    Returns:
        tokenizer (Tokenizer): Loaded tokenizer object.
        model (Model): Loaded Keras model.
    """
    
    # Load the tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Load the model
    model = load_model(model_path)
    
    return tokenizer, model

def preprocess_input(text: str, tokenizer, max_seq_length: int) -> np.ndarray:
    """
    Preprocess the input text to convert it into a padded sequence for the model.
    
    Args:
        text (str): The input text to be summarized.
        tokenizer (Tokenizer): Tokenizer object used for text processing.
        max_seq_length (int): Maximum length of sequences after padding.
        
    Returns:
        np.ndarray: Padded sequence of the input text.
    """
    
    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([text])
    
    # Pad sequence to fixed length
    padded_sequence = pad_sequences(sequence, maxlen=max_seq_length, padding='post', truncating='post')
    
    return padded_sequence

def generate_summary(input_text: str, tokenizer, model, max_seq_length: int) -> str:
    """
    Generate a summary for the given input text using the trained model.
    
    Args:
        input_text (str): The input text to be summarized.
        tokenizer (Tokenizer): Tokenizer object used for text processing.
        model (Model): The trained Keras model for text summarization.
        max_seq_length (int): Maximum length of sequences used in the model.
        
    Returns:
        str: The generated summary for the input text.
    """
    
    # Preprocess the input
    processed_input = preprocess_input(input_text, tokenizer, max_seq_length)
    
    # Generate the summary
    prediction = model.predict(processed_input)
    
    # Convert the prediction to text
    summary_sequence = np.argmax(prediction, axis=-1)
    summary = ' '.join(tokenizer.index_word[i] for i in summary_sequence[0] if i != 0)  # Exclude padding (index 0)
    
    return summary

def aggregate_summaries(summaries: List[str], original_texts: List[str]) -> str:
    """
    Aggregate summaries from multiple texts, reducing redundancy and ensuring a logical flow.
    
    Args:
        summaries (List[str]): List of summaries to be aggregated.
        original_texts (List[str]): Original texts corresponding to each summary.
        
    Returns:
        str: The final aggregated summary.
    """
    
    # Tokenize each summary into sentences
    all_sentences = [sent_tokenize(summary) for summary in summaries]
    flat_sentences = [sentence for sublist in all_sentences for sentence in sublist]  # Flatten the list
    
    # Calculate sentence similarity to remove redundancy
    vectorizer = CountVectorizer().fit_transform(flat_sentences)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)
    
    # Filter out highly similar sentences to reduce redundancy
    threshold = 0.9  # Similarity threshold, can be adjusted
    filtered_sentences = []
    for i in range(len(flat_sentences)):
        if i == 0 or all(cosine_matrix[i][j] < threshold for j in range(i)):
            filtered_sentences.append(flat_sentences[i])
    
    # Optionally, order sentences based on their appearance in the original texts
    ordered_sentences = sort_sentences(filtered_sentences, original_texts)
    
    # Concatenate the filtered sentences
    final_summary = ' '.join(ordered_sentences)
    return final_summary

def sort_sentences(filtered_sentences: List[str], original_texts: List[str]) -> List[str]:
    """
    Sort the filtered sentences based on their appearance in the original text.
    
    Args:
        filtered_sentences (List[str]): Sentences after redundancy removal.
        original_texts (List[str]): Original texts corresponding to each summary.
        
    Returns:
        List[str]: Ordered sentences based on the original text structure.
    """
    
    # Tokenize original texts into sentences
    original_sentences = [sent_tokenize(text) for text in original_texts]
    flat_original_sentences = [sentence for sublist in original_sentences for sentence in sublist]
    
    # Create a mapping of sentences to their index in the original text
    sentence_order = {sentence: idx for idx, sentence in enumerate(flat_original_sentences)}
    
    # Sort the filtered sentences based on their index in the original text
    ordered_sentences = sorted(filtered_sentences, key=lambda sentence: sentence_order.get(sentence, float('inf')))
    
    return ordered_sentences

def main():
    # Load model and tokenizer
    tokenizer, model = load_artifacts(TOKENIZER_PATH, MODEL_PATH)
    
    # Load or input your texts (for demonstration, we use a placeholder list of texts)
    texts = ['Text to summarize 1', 'Text to summarize 2']  # Replace with actual texts
    
    # Generate summaries and aggregate them
    summaries = [generate_summary(text, tokenizer, model, MAX_SEQ_LENGTH) for text in texts]
    final_summary = aggregate_summaries(summaries, texts)  # Provide original texts for ordering
    
    print(final_summary)

if __name__ == '__main__':
    main()