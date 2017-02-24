# Load our dataset
from sklearn.datasets import load_iris
#X, y = np.loadtxt("X_classification.txt"), np.loadtxt("y_classification.txt")
dataset = load_iris()
X = dataset.data
y = dataset.target
print(dataset.DESCR)
n_samples, n_features = X.shape