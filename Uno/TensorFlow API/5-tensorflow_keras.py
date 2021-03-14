import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from Uno.tools import *

# Official sample data for testing
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

split = 8000
validation_images = test_images[:split]
validation_labels = test_labels[:split]
test_images = test_images[split:]
test_labels = test_labels[split:]

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images, test_images, validation_images = train_images / 255.0, test_images / 255.0, validation_images / 255.0
print(train_images.shape) # the image is an array of images; each image is 28 x 28

# A basic neural network - creates the architecture
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # input, converts our 28 x 28 array into one vector
    keras.layers.Dense(128, activation="relu"), # a hidden layer using relu
    keras.layers.Dense(10, activation="softmax") # the output layer using softmax(the sum is equal to 1)
])
# add in the details to the network
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# model training
model.fit(train_images, train_labels, epochs=10)
print("Evaluated Accuracy: ", model.evaluate(validation_images, validation_labels, verbose=3)[1])

while True:
    thing = int(input("Please enter an index: "))
    if thing == -1:
        break
    possibilities = model.predict(np.array([test_images[thing]]))
    print("Model thinks this image is a: ", class_names[np.ndarray.argmax(possibilities)])

    print("Here was the model: ")
    plot_greyscale_image(test_images[thing])