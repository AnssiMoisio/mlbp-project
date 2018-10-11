from sklearn.cluster import KMeans, AffinityPropagation, MeanShift
import sklearn
import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.append('../')
from preprocessor import Preprocessor
from sklearn.cluster import KMeans
from keras.datasets import mnist
from sklearn.preprocessing import minmax_scale, StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, normalization, Input
from keras.optimizers import SGD
from keras import losses
from time import time
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.models import Model
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
import metrics
from clusteringlayer import ClusteringLayer
from autoencoder import Autoencoder

dl = Preprocessor(balance=True, path='../data/')
x, y, x_test, y_test = dl.divided_data(ratio=1, load_bal_data=True, save_bal_data=False)

y = y.reshape((len(y),))

# different data preprocessors:
# sklearn.preprocessing.normalize(x)
# minmax_scale(x)
# StandardScaler(x)

# 10 clusters
n_clusters = 10

# different clustering methods:
# clf = KMeans(n_clusters=n_clusters, n_init=20, max_iter=3000)
# clf = AffinityPropagation()
# clf = MeanShift()

# try basic clustering first for the data
kmeans = KMeans(n_clusters=n_clusters, n_init=20)
if __name__ == '__main__':
    y_pred = kmeans.fit_predict(x)

print("accuracy with kmeans: ", metrics.acc(y, y_pred))

# parameters for autoencoder
dims = [x.shape[-1], 200, 200, 2000, 10] # dims for the encoder layers, decoder is symmetric
init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
pretrain_optimizer = SGD(lr=1, momentum=0.9)
pretrain_epochs = 300
batch_size = 256
save_dir = './results'

autoencoder, encoder = Autoencoder(dims, init=init)

'''
# Train autoencoder
autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs) #, callbacks=cb)
autoencoder.save_weights(save_dir + '/ae_weights-200-200-2000-10-balanced.h5')
'''

# load pre-trained autoencoder
autoencoder.load_weights(save_dir + '/ae_weights-200-200-2000-10-balanced.h5')

if __name__ == '__main__':
    y_pred = kmeans.fit_predict(encoder.predict(x))

'''
# plot histogram of prediction
plt.figure(1, figsize=(10, 8))
plt.title("prediction label distribution")
plt.hist(y_pred, range=(-0.5,9.5), bins=10, ec='black')
plt.show()
'''

# Evaluate the K-Means clustering accuracy.
print("accuracy with encoder + kmeans: ", metrics.acc(y, y_pred))

# custom clustering layer
clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
model = Model(inputs=encoder.input, outputs=clustering_layer)
model.compile(optimizer=SGD(0.01, 0.9), loss='kld')
y_pred_last = np.copy(y_pred)
model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

# computing an auxiliary target distribution
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

loss = 0
index = 0
maxiter = 8000
update_interval = 140
index_array = np.arange(x.shape[0])

tol = 0.000000001 # tolerance threshold to stop training

'''
# train the model
for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q = model.predict(x, verbose=0)
        p = target_distribution(q)  # update the auxiliary target distribution p

        # evaluate the clustering performance
        y_pred = q.argmax(1)
        if y is not None:
            acc = np.round(metrics.acc(y, y_pred), 5)
            nmi = np.round(metrics.nmi(y, y_pred), 5)
            ari = np.round(metrics.ari(y, y_pred), 5)
            loss = np.round(loss, 5)
            print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

        # # check stop criterion - model convergence
        # delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        # y_pred_last = np.copy(y_pred)
        # if ite > 0 and delta_label < tol:
        #     print('delta_label ', delta_label, '< tol ', tol)
        #     print('Reached tolerance threshold. Stopping training.')
        #     break

    idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
    model.train_on_batch(x=x[idx], y=p[idx])
    index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

model.save_weights(save_dir + '/DEC_model_final.h5')
'''

# use pre-trained model
model.load_weights(save_dir + '/DEC_model_final.h5')

# Eval.
q = model.predict(x, verbose=0)
p = target_distribution(q)  # update the auxiliary target distribution p

# evaluate the clustering performance
y_pred = q.argmax(1)
if y is not None:
    acc = np.round(metrics.acc(y, y_pred), 5)
    nmi = np.round(metrics.nmi(y, y_pred), 5)
    ari = np.round(metrics.ari(y, y_pred), 5)
    loss = np.round(loss, 5)
    print('Acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari), ' ; loss=', loss)

