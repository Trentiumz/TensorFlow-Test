import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import models
import numpy as np

train = tfds.load("cats_vs_dogs", split="train[:80%]")
validation = tfds.load("cats_vs_dogs", split="train[80%:90%]")
test = tfds.load("cats_vs_dogs", split="train[90%:]")

train = train.map(lambda x: (tf.image.resize_with_pad(x["image"], 50, 50) / 255, x["label"]))
validation = validation.map(lambda x: (tf.image.resize_with_pad(x["image"], 50, 50) / 255, x["label"]))
test = test.map(lambda x: (tf.image.resize_with_pad(x["image"], 50, 50) / 255, x["label"]))
train = train.repeat().shuffle(buffer_size=1024).batch(64)
validation = validation.batch(64)
test = test.batch(64)

model = models.load_model("./model")
history = model.fit(train, steps_per_epoch=300, epochs=10, validation_data=validation)

print("EVALUATION")
loss, acc = model.evaluate(test, verbose=2)
print("loss: ", loss, ". accuracy: ", acc)

model.save("./model")