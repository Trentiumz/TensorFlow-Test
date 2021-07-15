import tensorflow as tf
from keras import models

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels, test_labels = train_labels.reshape((50000,)), test_labels.reshape((10000,))

model = models.load_model("./model")
print(model.input_shape)
model.fit(train_images, train_labels, batch_size=5000, epochs=10, validation_split=0.2)

print("EVALUATION")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("loss:", test_loss, "accuracy:", test_acc)

model.save("./model")
