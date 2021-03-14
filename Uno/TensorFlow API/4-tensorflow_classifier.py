from Uno.tools import *
from datetime import datetime
import pandas as pd

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv", names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv", names=CSV_COLUMN_NAMES, header=0)

train_ans = train.pop("Species")
test_ans = test.pop("Species")

features = get_features({}, [], CSV_COLUMN_NAMES[:4])
print(features)
classifier = make_classifier(features, [30, 10], 3)

train_func = get_classifier_input_fn(train, train_ans)
test_func = get_classifier_input_fn(test, test_ans, shuffle=False)

print("start training")
classifier.train(train_func, steps=10000)
print("finished training")

print("start evaluation")
result = classifier.evaluate(test_func)
print("finish evaluation")
print(result["accuracy"])
