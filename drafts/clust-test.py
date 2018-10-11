from sklearn.cluster import KMeans, AffinityPropagation, MeanShift
import sklearn
import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np
from preprocessor import Preprocessor
import metrics
from sklearn.cluster import KMeans
from keras.datasets import mnist
from sklearn.preprocessing import minmax_scale, StandardScaler


dl = Preprocessor(balance=False)
x_train, y_train, x_test, y_test = dl.divided_data(ratio=1)

x_train = np.delete(x_train, [0,1600], 0)
y_train = np.delete(y_train, [0,1600], 0)


# sklearn.preprocessing.normalize(x_train)
# minmax_scale(x_train)
StandardScaler(x_train)

# 10 clusters
n_clusters = 10

# Runs in parallel 4 CPUs, 20 initializations
# clf = KMeans(n_clusters=n_clusters, n_init=40, max_iter=3000)
# clf = AffinityPropagation()
clf = MeanShift()

if __name__ == '__main__':
    # Train K-Means.
    y_pred_kmeans = clf.fit_predict(x_train)

plt.figure(1, figsize=(10, 8))
plt.title("prediction label distribution")
plt.hist(y_pred_kmeans, range=(-0.5,9.5), bins=10, ec='black')
plt.show()

# Evaluate the K-Means clustering accuracy.
print(metrics.acc(y_train, y_pred_kmeans))
