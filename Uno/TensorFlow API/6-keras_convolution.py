import Uno.tools as tools
import tensorflow as tf
from tensorflow import keras as keras
import tensorflow_datasets as datasets
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# This is just getting data
(raw_train_images, raw_train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

split = 8000
validation_images = np.array(test_images[:split])
validation_labels = np.array(test_labels[:split])
test_images = np.array(test_images[split:])
test_labels = np.array(test_labels[split:])
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
raw_train_images = np.array(raw_train_images)
raw_train_labels = np.array(raw_train_labels)

# This allows us to generate new "versions" of the same image
datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                             shear_range=0.2, zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")

batch_size = 500
count = 400
# The generator is a function which will create new batches of batch_size with slightly altered images of raw_train_images
generator = datagen.flow(raw_train_images, raw_train_labels, batch_size=batch_size, shuffle=True)
# Store the images
train_images = np.zeros((count * batch_size, 32, 32, 3))
train_labels = np.zeros((count * batch_size, 1))
i = 0
# we essentially enumerate the generator like a for loop, but it's actually recalculating every time
# images is a 4d numpy array of batch_size images
for images, labels in generator:
    train_images[batch_size * i : batch_size * (i + 1)] = images
    train_labels[batch_size * i : batch_size * (i + 1)] = labels
    i += 1
    if i >= count:
        break
    print(i)

model = models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)), # (num of units, filter_shape)
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),

    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss=keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])

history = model.fit(train_images, train_labels, epochs=10, validation_data=(validation_images, validation_labels))
print(model.evaluate(test_images, test_labels))
model.save("Files/6_model.h5")

while True:
    thing = int(input("Please enter some index: "))
    if thing == -1:
        break
    print("model predicts ", class_names[np.ndarray.argmax(model.predict(np.array([test_images[thing]])))])
    tools.plot_color_image(test_images[thing])
