# FederatedLearningDemo
This project demonstrates simplified federated learning by extending scikit-learn's linear regression implementation to support federated averaging.

The extended implementation can be found in this repository at [federated_learning_demo/federated_linear_regression.py](federated_learning_demo/federated_linear_regression.py). This file defines the class `FederatedLinearRegression` as a subclass of `LinearRegression` from `sklearn.linear_model`. `FederatedLinearRegression` usage is exactly like `LinearRegression`; the key difference is that `FederatedLinearRegression` also includes a static method `federated_average`, which takes a list of `FederatedLinearRegression` models as an argument and returns a single `FederatedLinearRegression` model as the federated average of all the models from the list. Note that all models in the argument list for `federated_average` must be the same shape (i.e. have the same number of coefficients).

Example usage:
```
from federated_learning_demo.federated_linear_regression import FederatedLinearRegression

# Assume data_a, data_b, and data_c contain splits from the same dataset
data_a = ...
data_b = ...
data_c = ...

# Fit a new FederatedLinearRegression model separately for each dataset
flr_model_a = FederatedLinearRegression().fit(data_a.data, data_a.target)
flr_model_b = FederatedLinearRegression().fit(data_b.data, data_b.target)
flr_model_c = FederatedLinearRegression().fit(data_c.data, data_c.target)

# Create a list of federated models
federated_model_list = [ flr_model_a, flr_model_b, flr_model_c ]

# Call FederatedLinearRegression.federated_average, passing the list of federated models as the only argument
flr_averaged = FederatedLinearRegression.federated_average(federated_model_list)

# User the returned federated average model as you would normally
data_test = ...
flr_averaged.score(data_test.data, data_test.target)
```

For a working example using the Iris dataset, try [test_federated_learning.py](test_federated_learning.py). Clone this repository and run `python3 test_federated_learning.py`.