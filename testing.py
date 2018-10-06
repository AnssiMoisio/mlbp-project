from preprocessor import Preprocessor
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam

dl = Preprocessor(scale=False)
x_train, y_train, x_test, y_test = dl.divided_data(ratio=0.5)
y_train = dl.transformed_labels(y_train)
y_test = dl.transformed_labels(y_test)

opt = Adam(lr=1e-4)

model = Sequential()
model.add(Dense(400, activation='sigmoid', input_dim=x_train.shape[1]))
#model.add(Dropout(rate=0.5))
model.add(Dense(200, activation='sigmoid'))
#model.add(Dropout(rate=0.2))
model.add(Dense(16))
model.add(Dense(10, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=200,
          batch_size=1024)

score = model.evaluate(x_test, y_test, batch_size=128)
print("Score: ", score)