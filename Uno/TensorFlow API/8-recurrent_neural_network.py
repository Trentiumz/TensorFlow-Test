from tensorflow.keras.preprocessing import sequence
import tensorflow.keras as keras
import tensorflow as tf
import os
import numpy as np

# Get our text, and create the possible vocabulary
text = str(open("Files/Pride-and-Prejudice.txt", "rt").read())
print("Text Length =", len(text))

vocab = sorted(set(text))
VOCAB_COUNT = len(vocab)
char_to_int = {u: i for i, u in enumerate(vocab)}
int_to_char = list(vocab)

# utility functions for converting to and from encoded/text
def convert_to_int_arr(text):
    return [char_to_int[i] for i in text]
def convert_to_text(int_array):
    return "".join([int_to_char[i] for i in int_array])

encoded_text = convert_to_int_arr(text)

# Convert the text into a dataset, batch it, create the inputs/outputs from the known text, and finally seperate the test cases in batches
SEQ_LENGTH = 100
examples_per_epoch = len(text) // (SEQ_LENGTH + 1)
text = tf.data.Dataset.from_tensor_slices(encoded_text)
text = text.batch(SEQ_LENGTH + 1, drop_remainder=True)

text = text.map(lambda x: (x[:-1], x[1:])) # text is now ((start, end), (start, end),...)

BATCH_SIZE = 64
SHUFFLE_SIZE = 10000
text = text.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Create the actual model
EMBEDDING_DIM = 256
RNN_UNITS = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    # Embedding will convert straight characters into more useful information, by creating a multidimensional vector to group how "similar" characters are
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=(batch_size, None)),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform"),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(VOCAB_COUNT, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
print(model.summary())

# custom loss function
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
model.compile(optimizer="adam", loss=loss)

# Creating a checkpoint directory
checkpoint_dir = "Files/eight/checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
# Train the model, with a callback storing the current state after each epoch
print(text)
history = model.fit(text, epochs=100)