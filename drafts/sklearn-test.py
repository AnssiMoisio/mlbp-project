import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OutputCodeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
import sys
sys.path.append('../')
from preprocessor import Preprocessor

# from PCA import PCA

# labels = pd.read_csv("../data/train_labels.csv")
# training_data = pd.read_csv("../data/train_data.csv")

dl = Preprocessor(balance=False, scale=True, path='../data/')
x_train, y_train, x_test, y_test = dl.divided_data(ratio=0.8, load_bal_data=False)

# plot label distribution in training data
#plt.figure(1, figsize=(10, 8))
#plt.hist(labels.values, range=(0.5,10.5), bins=10, ec='black')

# suffle the data because it's ordered
# matrix = np.column_stack((labels, training_data))
# matrix = matrix[0:4362,]
# np.random.shuffle(matrix)
# y = matrix[:,0]
# training_data = np.delete(matrix, 0, 1)

# print(training_data[:10,0])
# pca = PCA(training_data, 100)
# training_data = pca.low_dim_data()

# divide into training data and validation data
# X = training_data[:3000,]
# y1 = y[:3000]
# X2 = training_data[3000:,]
# y2 = y[3000:]

clf = LogisticRegression(random_state=0, solver='saga',multi_class='multinomial', max_iter=1000, verbose=1)
# clf = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=1000, verbose=1)) 
# clf = OutputCodeClassifier(LinearSVC(random_state=0, max_iter=1000, verbose=1), code_size=2, random_state=0)
# clf = linear_model.SGDClassifier(max_iter=300, loss='log', verbose=1)

model = clf.fit(x_train, y_train)
prediction = model.predict(x_test)

# plot label distribution in prediction
print("train score: ", clf.score(x_train, y_train))
print("test score: ", clf.score(x_test, y_test))




'''
test_data = dl.test_data
prediction = model.predict(test_data)
pd.DataFrame(prediction).to_csv('result.csv')

plt.figure(2, figsize=(10, 8))
plt.title("prediction label distribution")
plt.hist(prediction, range=(0.5,10.5), bins=10, ec='black')
plt.show()

data = pd.read_csv("../data/test_data.csv")
sklearn.preprocessing.normalize(data)
pred = model.predict(data)
# y_classes = pred.argmax(axis=-1) # convert probabilities into labels

plt.figure(2, figsize=(10, 8))
plt.title("prediction label distribution")
plt.hist(pred, range=(-0.5,9.5), bins=10, ec='black')
plt.show()

# y_classes = np.subtract(y_classes, [-1]*6544)
pd.DataFrame(pred).to_csv('result-1.csv')
'''