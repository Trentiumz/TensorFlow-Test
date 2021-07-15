import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from keras import models
import matplotlib.pyplot as plt

class_name = ["cat", "dog"]

def plot_image(img, lab):
    plt.figure(figsize=(50, 50))
    plt.grid(False)
    plt.imshow(img)
    plt.xlabel("Animal: " + class_name[lab])
    plt.show()

model = models.load_model("./model")
train = tfds.load("cats_vs_dogs")["train"]
for i in train:
    img = tf.image.resize_with_pad(i["image"], 50, 50).numpy().reshape((1, 50, 50, 3)) / 255
    pred = model(img).numpy()[0][0]
    plot_image(img.reshape((50, 50, 3)), round(pred))
    if input() == "exit":
        break