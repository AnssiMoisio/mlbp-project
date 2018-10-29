import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OutputCodeClassifier
import sys
sys.path.append('../')
from PCA import PCA

labels = pd.read_csv("../data/train_labels.csv")
training_data = pd.read_csv("../data/train_data.csv")

# plot label distribution in training data
#plt.figure(1, figsize=(10, 8))
#plt.hist(labels.values, range=(0.5,10.5), bins=10, ec='black')

# suffle the data because it's ordered
matrix = np.column_stack((labels, training_data))
matrix = matrix[1600:4362,]
np.random.shuffle(matrix)
y = matrix[:,0]
training_data = np.delete(matrix, 0, 1)

sklearn.preprocessing.normalize(training_data)
pca = PCA(training_data, 100)
training_data = pca.low_dim_data()

# divide into training data and validation data
X = training_data[:2000,]
y1 = y[:2000]
X2 = training_data[2000:,]
y2 = y[2000:]

clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial') # 53%
#clf = OneVsOneClassifier(LinearSVC(random_state=0)) # 19 - 57 %
#clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0) # 37 - 52 %

model = clf.fit(X,y1)
prediction = model.predict(X2)

# plot label distribution in prediction
print("score: ", clf.score(X2, y2))


'''
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