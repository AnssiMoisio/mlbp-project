import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sklearn.preprocessing

from sklearn.linear_model import LogisticRegression

class PCA:
    def __init__(self, data, d):
        self.data = data # raw data
        self.d = d # number of dimensions in the compressed data
        self.W_pca, self.eigvalues = self.compute_pca()

    def compute_pca(self):
        # Output: a d by D matrix W_pca, and all eigenvalues of Q

        N = self.data.shape[0]
        
        # step1: compute the sample cov. matrix Q
        Q = np.matmul(np.transpose(self.data), self.data ) / N

        #step2: compute the eigenvalues and eigenvectors (see introduction notebook)
        w, v = np.linalg.eig(Q)
        
        #step3: Sort the eigenvectors by decreasing eigenvalues, choose the d largest eigenvalues, form W_pca
        ind = np.argsort(w)[::-1]
        W_pca = np.empty((self.d, self.data.shape[1]))
        eigvalues = w
        for i in range(self.d):
            W_pca[i] = v[:,ind[i]]
        
        return W_pca.real, eigvalues # discard imaginary part

    def plot_error(self,  max_d):
        x=range(1,max_d+1)
        errors=[sum(self.eigvalues[d:]) for d in x]
        plt.plot(x,errors)
        plt.xlabel('Number of principal components $d$')
        plt.ylabel('Reconstruction error $\mathcal{E}$')
        plt.title('Number of principal components vs the reconstruction error')
        plt.show()

    def plot_scatter(self):
        # get x for d=2
        X_2d = np.matmul(self.W_pca[:2,:],self.data[:,:,None])[:,:,0]
        plt.figure(1, figsize=(10, 10))
        plt.scatter(X_2d[:,0],X_2d[:,1], 3)
        plt.legend()
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')
        plt.show()
    
    def low_dim_data(self):
        new_data = np.ndarray((self.data.shape[0], self.d))
        for i in range(self.data.shape[0]):
            new_data[i] = np.matmul(self.W_pca, self.data[i])
        return new_data
        

# def main():
#     data = pd.read_csv('/data/train_data.csv')
#     data = sklearn.preprocessing.normalize(data)
#     pca = PCA(data, 50)

#     print(pca.low_dim_data().shape)
  
#     # plot the number of principal components vs the reconstruction error
#     pca.plot_error(30)

#     pca.plot_scatter()

# main()