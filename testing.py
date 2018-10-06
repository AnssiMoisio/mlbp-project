from preprocessor import Preprocessor
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

data_loader = Preprocessor()

x_train, y_train, x_test, y_test = data_loader.divide_data()

model = Sequential()
model.add(Dense(64, input_dim=training_data.shape[1]))