# Seed for reproducibility in random operations
SEED = 42

# Text preprocessing configurations
LOWER_CASE = False  # If True, convert all text to lower case

# LSTM Model configurations
LATENT_DIM = 256  # Dimensionality of the hidden state in the LSTM

# Data chunking configurations for handling large texts
CHUNK_THRESHOLD = 500  # Maximum number of words in a text before it is chunked

# Sequence configurations
MAX_SEQ_LENGTH = 100  # Maximum sequence length for padding
VOCAB_SIZE = 10000    # Size of the vocabulary

# Embedding configurations
EMBEDDING_DIM = 100   # Dimension of word vectors in GloVe embeddings
GLOVE_PATH = 'data/glove.6B.100d.txt'  # Filepath to the GloVe embeddings

# Training configurations
BATCH_SIZE = 64       # Batch size for training
EPOCHS = 100          # Number of epochs to train for
LEARNING_RATE = 0.001 # Learning rate for the optimizer

# Filepath configurations for saving and loading model components
CHECKPOINT_PATH = 'models/best_model.h5'            # Filepath for saving the best model during training
TOKENIZER_PATH = 'models/tokenizer.pkl'             # Filepath for saving the tokenizer
TRAINING_HISTORY_PATH = 'data/training_history.pkl' # Filepath for saving training history
TRAINING_DATA_PATH = 'data/training/'               # Filepath for training data
MODEL_PATH = 'models/best_model.h5'                 # Filepath for the trained model

