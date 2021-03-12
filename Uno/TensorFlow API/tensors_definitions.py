from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import pandas as pd

print(tf.version)
print()


def variables_test():
    string = tf.Variable("this is a string", tf.string)
    integer = tf.Variable(324, tf.int16)
    floating = tf.Variable(123.123, tf.float32)
    print("Variable Test")
    print(f"{string} {integer} {floating}")


def ranks_test():
    # Tensor Ranks/Degrees are # of dimensions
    # Tensors are basically a Vector/Matrix but can have even more dimensions
    rank1_tensor = tf.Variable([1, 2, 3, 4], tf.int8)
    rank2_tensor = tf.Variable([[1, 2, 3], [1, 2, 3]], tf.int8)
    print()
    print("RANKS")
    print(
        f"rank2_tensor has {tf.rank(rank2_tensor)} dimensions of {'*'.join([str(x) for x in rank2_tensor.shape])} shape")
    print(
        f"rank1_tensor has {tf.rank(rank1_tensor)} dimensions of {'*'.join([str(x) for x in rank1_tensor.shape])} shape")
    print()


def reshaping_test():
    rank2_tensor = tf.Variable([[1, 2, 3], [1, 2, 3]], tf.int8)
    # Reshaping will first flatten the entire tensor and then rebuild it
    print()
    print("RESHAPING")
    flat_r2 = tf.reshape(rank2_tensor, [6])
    print(flat_r2)
    rank2_tensor_2 = tf.reshape(rank2_tensor, [3, 2])
    print(rank2_tensor_2)
    flat_r2 = tf.reshape(rank2_tensor, [-1])  # putting -1 tells the machine to automatically fill in the value
    print(flat_r2)
    print()


def pandas_test():
    dftrain = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
    dfeval = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/eval.csv")
    print()
    print("PANDAS TEST")
    print("dftrain before removal: \n", dftrain)
    print()
    y_train = dftrain.pop("survived")
    y_eval = dfeval.pop("survived")
    print()
    print("y_train at head: \n", y_train.head())
    print()
    print("The two variables at row 0: \n", dftrain.loc[0], y_train.loc[0])
    print()
    print("describing: \n", dftrain.describe())
    print()
    print("shape: ", dftrain.shape)
    print()


variables_test()
ranks_test()
reshaping_test()
pandas_test()
