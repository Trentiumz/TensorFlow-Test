import tensorflow as tf
from tensorflow import keras
from keras import models, layers

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(50, 50, 3)),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation="relu"),
    layers.Dense(1, activation="sigmoid")])

model.compile(optimizer="adam", loss=tf.losses.BinaryCrossentropy(from_logits=False), metrics=["accuracy"])

model.save("./model")