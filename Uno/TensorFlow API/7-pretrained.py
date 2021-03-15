import Uno.tools as tools
import tensorflow as tf
from tensorflow import keras as keras
import tensorflow_datasets as datasets
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds
import numpy as np

(raw_train, raw_validation, raw_test), metadata = tfds.load("cats_vs_dogs",
                                                            split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
                                                            with_info=True,
                                                            as_supervised=True)  # this split is just normal arrays
raw_test = raw_test.shuffle(1000)
get_label = metadata.features["label"].int2str
IMG_SIZE = 160
def format_image(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 127.0 - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

model = tf.keras.models.load_model("Files/7_model.h5")

for image, label in raw_test.take(100000):
    q = input("quit? ")
    if q != "no":
        break
    im2, label = format_image(image, label)
    result = model.predict(np.array(im2).reshape((1,160,160,3))).reshape(-1)
    print("model predicts", get_label(round(result[0])))
    print("this was actuallly a", get_label(np.array(label)))
    tools.plot_color_image(image)