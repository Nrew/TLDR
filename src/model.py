"""Module to create an LSTM-based model for text summarization."""

from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from keras.layers import Attention
from constants import LATENT_DIM, EMBEDDING_DIM

def build_encoder(vocab_size, max_seq_length):
    """
    Construct the encoder component of the sequence-to-sequence model.

    The encoder processes the input sequence and compresses the information into 
    a context vector used by the decoder as the initial hidden state.

    Args:
        vocab_size (int): The size of the vocabulary, defining the number of unique tokens.
        max_seq_length (int): The maximum length of sequences the encoder will process.

    Returns:
        encoder_inputs (Tensor): The input layer of the encoder.
        encoder_outputs (Tensor): The output of the LSTM layer.
        encoder_states (list): LSTM state (hidden and cell state) after processing the input sequence.
    """
    
    encoder_inputs = Input(shape=(max_seq_length,))
    
    # Embedding layer converts token indices to dense vectors
    encoder_embedding = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=True)(encoder_inputs)
    
    # LSTM layer to process the sequence
    encoder_lstm = LSTM(LATENT_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    
    # Capturing the internal state of the LSTM to use in the decoder
    encoder_states = [state_h, state_c]
    
    return encoder_inputs, encoder_outputs, encoder_states

def build_decoder(vocab_size, max_seq_length, encoder_outputs, encoder_states):
    """
    Construct the decoder component of the sequence-to-sequence model.

    The decoder is responsible for generating the output sequence. It is initialized 
    with the context vector from the encoder (encoder_states), and it uses attention to focus 
    on specific parts of the input sequence.

    Args:
        vocab_size (int): The size of the vocabulary.
        max_seq_length (int): The maximum length of sequences the decoder will generate.
        encoder_outputs (Tensor): The output of the encoder.
        encoder_states (list): The state (hidden and cell state) of the encoder.

    Returns:
        decoder_inputs (Tensor): The input layer of the decoder.
        dense_time (Tensor): The output of the decoder after the TimeDistributed Dense layer.
    """
    
    # Define decoder input
    decoder_inputs = Input(shape=(max_seq_length,))
    
    # Embedding layer
    decoder_embedding = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=True)(decoder_inputs)
    
    # LSTM layer
    decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    
    # Attention layer to focus on relevant parts of the input sequence
    attention_layer = Attention()
    attention_result = attention_layer([decoder_outputs, encoder_outputs])
    
    # Concatenating the output of the attention layer with the output of the decoder LSTM
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attention_result])
    
    # TimeDistributed Dense layer to generate probabilities over the vocabulary for each time step
    dense = TimeDistributed(Dense(vocab_size, activation='softmax'))
    dense_time = dense(decoder_concat_input)
    
    return decoder_inputs, dense_time

def build_model(vocab_size, max_seq_length):
    """
    Build and compile the sequence-to-sequence model for text summarization.

    Args:
        vocab_size (int): The size of the vocabulary.
        max_seq_length (int): The maximum length of sequences.

    Returns:
        model (Model): The compiled Keras model ready for training.
    """
    
    # Construct encoder and decoder components
    encoder_inputs, encoder_outputs, encoder_states = build_encoder(vocab_size, max_seq_length)
    decoder_inputs, decoder_dense_time = build_decoder(vocab_size, max_seq_length, encoder_outputs, encoder_states)
    
    # Define the model taking encoder_inputs and decoder_inputs and outputting decoder_dense_time
    model = Model([encoder_inputs, decoder_inputs], decoder_dense_time)
    
    # Compile the model with adam optimizer and categorical crossentropy loss
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
    