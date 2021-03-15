import Uno.tools as tools
import tensorflow as tf
from tensorflow import keras as keras
import tensorflow_datasets as datasets
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds
import numpy as np

# loading some images in - these things are datasets in form [(image, label), (image, label), ...]
(raw_train, raw_validation, raw_test), metadata = tfds.load("cats_vs_dogs",
                                                            split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
                                                            with_info=True,
                                                            as_supervised=True)  # this split is just normal arrays
get_label_name = metadata.features["label"].int2str  # a function which gets the label name of an integer

IMG_SIZE = 160


def format_image(image, label):
    image = tf.cast(image, tf.float32)
    # puts the image between -1 and 1
    image = (image / 127.5) - 1
    # resizes the image to 160x160(by 3, but the 3 channels are a property of an image)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

# map each element to their formatted version
train = raw_train.map(format_image)
validation = raw_validation.map(format_image)
test = raw_test.map(format_image)

# seperate the data into batches, and shuffle training data
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

# copy the base model from tensorflow
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
base_model.trainable = False

# import the base model into a new generic model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# add compilation
learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=["accuracy"])

# actually train the model
history = model.fit(train_batches, epochs=3, validation_data = validation_batches)
print(history.history["accuracy"])

model.save("Files/7_model.h5")

for image, label in raw_test:
    a = input("quit?")
    if a != "no":
        break
    print("The AI thinks this is a", get_label_name(round(model.predict(format_image(image, label)[0]))))
    print("This really was a", get_label_name(label))
    tools.plot_color_image(image)