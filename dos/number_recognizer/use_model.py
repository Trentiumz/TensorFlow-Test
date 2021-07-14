from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

model = keras.models.load_model("./model")

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

def show(i: int):
    plt.imshow(x_test[i])
    print(y_test[i])
    result = model(x_test[i].reshape(1, 28, 28), training=False).numpy()
    print(result)
    print(np.argmax(result))
    plt.show()

print("Model Evaluation: ", model.evaluate(x_test, y_test))

for ind in range(len(x_test)):
    show(ind)
    input()