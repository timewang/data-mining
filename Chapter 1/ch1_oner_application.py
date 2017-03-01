import numpy as np
# Load our dataset
from sklearn.datasets import load_iris
#X, y = np.loadtxt("X_classification.txt"), np.loadtxt("y_classification.txt")
dataset = load_iris()
X = dataset.data
# 目标分类，0、1、2 三个分类
y = dataset.target
print(dataset)
#print(dataset.DESCR)
# 150 * 4 数据共有这么多
n_samples, n_features = X.shape

# Compute the mean for each attribute 每一列数据的平均值
attribute_means = X.mean(axis=0)
print(attribute_means)
assert attribute_means.shape == (n_features,)
# 按照上面算出的阈值进行分类，上面算出的阈值为：[ 5.84333333  3.054       3.75866667  1.19866667]。计算每一列每个值与对应的阈值进行比较
# 大于阈值的为 1 小于阈值的为 0
X_d = np.array(X >= attribute_means, dtype='int')
print( X_d.shape)
# Now, we split into a training and test set
from sklearn.cross_validation import train_test_split

# Set the random state to the same number to get the same results as in the book
random_state = 14
# 从样本中按比例选取 train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_d, y, random_state=random_state)
print(X_train.shape)
print(X_test.shape)
print("There are {} training samples".format(y_train.shape))
print("There are {} testing samples".format(y_test.shape))

from collections import defaultdict
from operator import itemgetter


def train(X, y_true, feature):
    """Computes the predictors and error for a given feature using the OneR algorithm

    Parameters
    ----------
    X: array [n_samples, n_features]
        The two dimensional array that holds the dataset. Each row is a sample, each column
        is a feature.类别数组

    y_true: array [n_samples,]
        The one dimensional array that holds the class values. Corresponds to X, such that
        y_true[i] is the class value for sample X[i].

    feature: int
        An integer corresponding to the index of the variable we wish to test.
        0 <= variable < n_features 第几列数据，既第几个特征值

    Returns
    -------
    predictors: dictionary of tuples: (value, prediction)
        For each item in the array, if the variable has a given value, make the given prediction.

    error: float
        The ratio of training data that this rule incorrectly predicts.
    """
    # Check that variable is a valid number
    n_samples, n_features = X.shape
    assert 0 <= feature < n_features
    # Get all of the unique values that this variable has
    # 以数组形式返回由feature_index所指的列的值.然后以set函数将数组转为集合.
    values = set(X[:, feature])
    # Stores the predictors array that is returned
    predictors = dict()
    errors = []
    for current_value in values:
        most_frequent_class, error = train_feature_value(X, y_true, feature, current_value)
        predictors[current_value] = most_frequent_class
        errors.append(error)
    # Compute the total error of using this feature to classify on
    total_error = sum(errors)
    # ，返回预测器及总错误率。predictors 每一列阈值，所对应的类别
    return predictors, total_error


# Compute what our predictors say each sample is based on its value
# y_predicted = np.array([predictors[sample[feature]] for sample in X])

#，参数分别是数据集、类别数组、选好的特征索引值、特征值。
# 遍历数据集中每一条数据，统计具有给定特征值的个体在各个类别中的出现次数
def train_feature_value(X, y_true, feature, value):
    # Create a simple dictionary to count how frequency they give certain predictions
    # 每个类别出现的次数
    class_counts = defaultdict(int)
    # Iterate through each sample and count the frequency of each class/value pair
    # zip 方法，接收两个参数，都是 array，循环每个值，作为 key 和 value，这里 key 为样本数据，value 为 响应的类别
    # zip 用法
    for sample, y in zip(X, y_true):
        if sample[feature] == value:
            class_counts[y] += 1
    # Now get the best one by sorting (highest first) and choosing the first item
    sorted_class_counts = sorted(class_counts.items(), key=itemgetter(1), reverse=True)
    most_frequent_class = sorted_class_counts[0][0]# 频率最高的类别 此处是 0
    # The error is the number of samples that do not classify as the most frequent class
    # *and* have the feature value.
    n_samples = X.shape[1]
    # 没有被归类为最高类别的
    error = sum([class_count for class_value, class_count in class_counts.items()
                 if class_value != most_frequent_class])
    return most_frequent_class, error

# Compute all of the predictors 训练数据，返回结果是个 map key 就是 列 编号，value
all_predictors = {variable: train(X_train, y_train, variable) for variable in range(X_train.shape[1])}
errors = {variable: error for variable, (mapping, error) in all_predictors.items()}
# Now choose the best and save that as "model"
# Sort by error,
best_variable, best_error = sorted(errors.items(), key=itemgetter(1))[0]
print("The best model is based on variable {0} and has error {1:.2f}".format(best_variable, best_error))

# Choose the bset model 这里值是 2，就是说第三列的错误率最小，以第三列的分类为准 {'variable': 2, 'predictor': {0: 0, 1: 2}}
# 这个规则是，这列的值如果为0 分类就是0 ，如果为1 分类就是 2
model = {'variable': best_variable,
         'predictor': all_predictors[best_variable][0]}# 取错误率最小的规则
print(model)

def predict(X_test, model):
    variable = model['variable']
    predictor = model['predictor']
    # 取第三列的值进行分类
    y_predicted = np.array([predictor[int(sample[variable])] for sample in X_test])
    print(y_predicted)
    return y_predicted

y_predicted = predict(X_test, model)
print(y_predicted)

# Compute the accuracy by taking the mean of the amounts that y_predicted is equal to y_test
# 通过算法得出的分类与实际分类进行比较，得到算法的准确率
accuracy = np.mean(y_predicted == y_test) * 100
print("The test accuracy is {:.1f}%".format(accuracy))

from sklearn.metrics import classification_report

print(classification_report(y_test, y_predicted))
