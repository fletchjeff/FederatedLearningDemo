from federated_learning_demo.federated_linear_regression import FederatedLinearRegression
import numpy as np
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load Iris dataset
iris = datasets.load_iris()

# Split Iris dataset (to simulate a federated dataset; not for train/test)
iris_split_A_data, iris_split_B_data, iris_split_A_target, iris_split_B_target = train_test_split(iris.data, iris.target, test_size=0.30, random_state=30)

# Build model A
lrA = FederatedLinearRegression()
lrA.fit(iris_split_A_data, iris_split_A_target)

print('Model A:')
print("\tNum Samples - {}".format(lrA.num_samples_))
print("\tR2 - {}".format(lrA.score(iris.data, iris.target)))
print("\tCoefficients - {}".format(lrA.coef_))
print("\tIntercept - {}".format(lrA.intercept_))

# Save model A
model_A_file = "/tmp/model_A.pkl"
pickle.dump(lrA, open(model_A_file, 'wb'))

# Build model B
lrB = FederatedLinearRegression()
lrB.fit(iris_split_B_data, iris_split_B_target)

print('Model B:')
print("\tNum Samples - {}".format(lrB.num_samples_))
print("\tR2 - {}".format(lrB.score(iris.data, iris.target)))
print("\tCoefficients - {}".format(lrB.coef_))
print("\tIntercept - {}".format(lrB.intercept_))

# Save model B
model_B_file = "/tmp/model_B.pkl"
pickle.dump(lrB, open(model_B_file, 'wb'))

# Load saved models
lrA_from_disk = pickle.load(open(model_A_file, 'rb'))
lrB_from_disk = pickle.load(open(model_B_file, 'rb'))

# Create a new model as the average of models A and B
flr_list = [lrA_from_disk, lrB_from_disk]
iris_lr_fed_avg = FederatedLinearRegression.federated_average(flr_list)

print('Federated Average Model:')
print("\tNum Samples - {}".format(iris_lr_fed_avg.num_samples_))
print("\tR2 - {}".format(iris_lr_fed_avg.score(iris.data, iris.target)))
print("\tCoefficients - {}".format(iris_lr_fed_avg.coef_))
print("\tIntercept - {}".format(iris_lr_fed_avg.intercept_))