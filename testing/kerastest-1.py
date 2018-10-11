import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, normalization
from keras.optimizers import SGD
from keras import losses

labels = pd.read_csv("../data/train_labels.csv", index_col=False)
training_data = pd.read_csv("../data/train_data.csv",index_col=False)

# suffle the data because it's ordered
matrix = np.column_stack((labels, training_data))
matrix = matrix[1600:4362,:] # reduce the dominance of poprock in the music genre distribution
np.random.shuffle(matrix)
y_train = matrix[:,0]
x_train = np.delete(matrix, 0, 1)

# shift labels from 1-10 to 0-9
y_train = np.subtract(y_train, [1]*2762)

plt.figure(1, figsize=(10, 8))
plt.title("training data label distribution")
plt.hist(y_train, range=(-0.5,10.5), bins=10, ec='black')


sgd = SGD(lr=1, decay=1e-6, momentum=0.9, nesterov=True)
model = Sequential()
model.add(normalization.BatchNormalization())
model.add(Dense(10, activation='relu',input_dim=264))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer=sgd,
              loss=losses.mean_squared_error,
              metrics=['accuracy'])

# convert labels to correct form
y_train = keras.utils.to_categorical(y_train, num_classes=10)

# train
model.fit(x_train, y_train,
          validation_split=0.5, # split to  training and validation data 
          epochs=10, # there is over-fitting when epochs > 10
          batch_size=30)

# note that this prediction is done to whole x_train and not just validation data
prediction = model.predict(x_train,verbose=2)
y_classes = prediction.argmax(axis=-1) # convert probabilities into labels

plt.figure(2, figsize=(10, 8))
plt.title("prediction label distribution")
plt.hist(y_classes, range=(-0.5,9.5), bins=10, ec='black')

