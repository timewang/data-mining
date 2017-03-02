# import os
# home_folder = os.path.expanduser("~")
# print(home_folder)

# Change this to the location of your dataset
# data_folder = os.path.join(home_folder, "Data", "Ionosphere")
# data_filename = os.path.join(data_folder, "ionosphere.data")
# print(data_filename)

import csv
import numpy as np

# Size taken from the dataset and is known
X = np.zeros((351, 34), dtype='float')
y = np.zeros((351,), dtype='bool')

with open('ionosphere.data', 'r') as input_file:
    reader = csv.reader(input_file)
    for i, row in enumerate(reader):
        # Get the data, converting each item to a float
        data = [float(datum) for datum in row[:-1]]
        # Set the appropriate row in our dataset
        X[i] = data
        # 1 if the class is 'g', 0 otherwise
        y[i] = row[-1] == 'g'

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14)
print("There are {} samples in the training dataset".format(X_train.shape[0]))
print("There are {} samples in the testing dataset".format(X_test.shape[0]))
print("Each sample has {} features".format(X_train.shape[1]))

# 近邻算法
from sklearn.neighbors import KNeighborsClassifier

estimator = KNeighborsClassifier()

# 训练数据
estimator.fit(X_train, y_train)

# 测试
y_predicted = estimator.predict(X_test)
accuracy = np.mean(y_test == y_predicted) * 100
print("The accuracy is {0:.1f}%".format(accuracy))

# 交叉检验
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(estimator, X, y, scoring='accuracy')
average_accuracy = np.mean(scores) * 100
print("The average accuracy is {0:.1f}%".format(average_accuracy))