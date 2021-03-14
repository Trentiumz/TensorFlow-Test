import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets
import Uno.tools as tools

model = tf.keras.models.load_model("Files/6_model.h5")

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

split = 8000
validation_images = test_images[:split]
validation_labels = test_labels[:split]
test_images = test_images[split:]
test_labels = test_labels[split:]
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

while True:
    thing = int(input("Please enter some index: "))
    if thing == -1:
        break
    print("model predicts", class_names[np.ndarray.argmax(model.predict(np.array([test_images[thing]])))])
    print("this was actually a", class_names[test_labels[thing][0]])
    tools.plot_color_image(test_images[thing])