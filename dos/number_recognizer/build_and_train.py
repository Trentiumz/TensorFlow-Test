import tensorflow as tf
from tensorflow import keras
from keras import layers

model = keras.Sequential()
model.add(keras.Input(shape=(28, 28), name="Input"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu", name="Dense_1"))
model.add(layers.Dense(64, activation="relu", name="Dense_2"))
model.add(layers.Dense(10, activation="softmax", name="prediction"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255
x_val, y_val = x_train[-10000:], y_train[-10000:]
x_train, y_train = x_train[:-10000], y_train[:-10000]

tf_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
tf_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
tf_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

tf_train = tf_train.shuffle(buffer_size=1024).batch(64, drop_remainder=True)
tf_test = tf_test.shuffle(buffer_size=1024).batch(64, drop_remainder=True)
tf_val = tf_val.shuffle(buffer_size=1024).batch(64, drop_remainder=True)

model.fit(tf_train, epochs=8, validation_data=tf_val)

print("MODEL EVALUATION")
print(model.evaluate(tf_test))

model.save("./model")
