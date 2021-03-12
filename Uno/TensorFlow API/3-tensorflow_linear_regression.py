from Uno.tools import *
import numpy as np

elements = 500
thing = {"bob": list(range(elements))}
output = [1 if x > 200 else 0 for x in range(elements)]

# dftrain = pd.read_csv("./thing.csv")
# y_train = dftrain.pop("ByThree")

features = get_features(thing, [], ["bob"])
regressor = make_linear_regressor(features)
input_func = make_input_fn_from_basic(thing, output)

randoms = list(np.random.randint(0, 5000, 300))
eval_func = make_input_fn_from_basic({"bob": randoms}, [1 if x > 200 else 0 for x in randoms], batch_size=300,
                                     num_epochs=1, shuffle=False)
print(input_func())

regressor.train(input_func)
result = regressor.evaluate(eval_func)
print(result["accuracy"])