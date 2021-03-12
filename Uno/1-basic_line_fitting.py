import numpy as np
from matplotlib import pyplot as plt

w_learning_rate = 0.0001
epochs = 2000
my_batch_size = 50
b_learning_rate = w_learning_rate * my_batch_size

my_feature = range(my_batch_size)
my_label = np.array([-3 * x for x in range(my_batch_size)]) + (np.random.random(my_batch_size) * 100 - 50) + 50


# [w, b]
def build_model():
    return [0, 0]


def train_model(model):
    errors = []
    for i in range(epochs):
        # plot_model(model)
        total_w_grad = 0
        total_b_grad = 0
        total_loss = 0
        for index in range(my_batch_size):
            x = my_feature[index]
            y = my_label[index]

            total_loss += (y - model[1] - model[0] * x)**2

            total_w_grad += 2 * (y - model[1] - model[0] * x) * x
            total_b_grad += 2 * (y - model[1] - model[0] * x)
        model[0] += total_w_grad / my_batch_size * w_learning_rate
        model[1] += total_b_grad / my_batch_size * b_learning_rate
        # print(f"{total_loss} {total_w_grad} {total_b_grad} {total_w_grad / my_batch_size} {total_b_grad / my_batch_size}")
        errors.append(total_loss)
        # plot_model(model)
    return range(epochs), errors


def plot_model(model):
    plt.xlabel("feature")
    plt.ylabel("label")

    print(model)
    plt.scatter(my_feature, my_label)
    x0 = 0
    xf = max(my_feature)
    y0 = model[1]
    yf = xf * model[0] + model[1]
    plt.plot([x0, xf], [y0, yf], c='r')

    plt.show()


def plot_loss(epochs, rmse):
    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.plot(epochs, rmse)
    plt.show()


my_model = build_model()
epoc, rmse = train_model(my_model)

plot_model(my_model)
plot_loss(epoc, rmse)
