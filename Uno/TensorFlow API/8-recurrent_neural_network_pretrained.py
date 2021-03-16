from tensorflow.keras.preprocessing import sequence
import tensorflow.keras as keras
import tensorflow as tf
import os
import numpy as np

text = str(open("Files/Pride-and-Prejudice.txt", "rt").read())
print("Text Length =", len(text))

EMBEDDING_DIM = 256
RNN_UNITS = 1024

vocab = sorted(set(text))
VOCAB_COUNT = len(vocab)
char_to_int = {u: i for i, u in enumerate(vocab)}
int_to_char = list(vocab)

def convert_to_int_arr(text):
    return [char_to_int[i] for i in text]
def convert_to_text(int_array):
    return "".join([int_to_char[i] for i in int_array])

model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_COUNT, EMBEDDING_DIM, batch_input_shape=(1, None)),
        tf.keras.layers.LSTM(RNN_UNITS, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform"),
        tf.keras.layers.Dense(VOCAB_COUNT)
    ])
model.load_weights("Files/eight/ckpt")

TEXT_LENGTH = 1000
def generate_text(starting_text):
    input_eval = convert_to_int_arr(starting_text)
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []

    # lower more predictable, higher more risque
    temperature = 0.001

    model.reset_states()
    for i in range(TEXT_LENGTH):
        # the model keeps the state after adding each of the things
        predictions = model(input_eval)[0]
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(convert_to_text([predicted_id]))
    return starting_text + "".join(text_generated)

text = generate_text(input("Please enter some text: "))
print("Here is your story, enjoyyyy")
print("=" * 500)
print(text)