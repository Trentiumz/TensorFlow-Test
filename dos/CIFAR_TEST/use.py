import tensorflow as tf
from tensorflow import keras, data
from keras import models
import matplotlib.pyplot as plt
import numpy as np

(train_img, train_lab), (test_img, test_lab) = (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
test_img = test_img / 255.0
test_lab = test_lab.reshape((10000,))

dataset = data.Dataset.from_tensor_slices((test_img, test_lab))
dataset = dataset.repeat().shuffle(buffer_size=1000)


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def show(img, true, pred):
    plt.figure(figsize=(10, 10))
    plt.grid(False)
    plt.imshow(img)
    plt.xlabel("True: " + class_names[true] + "; Predict: " + class_names[pred])
    plt.show()

model = models.load_model("./model")
for imT, laT in dataset:
    img = imT.numpy()
    lab = int(laT.numpy())
    result = model(img.reshape((1, 32, 32, 3))).numpy()
    show(img, lab, np.argmax(result))
    if input() == "exit":
        break