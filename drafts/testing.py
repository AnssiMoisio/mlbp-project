from preprocessor import Preprocessor
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, normalization
from keras.optimizers import SGD, Adam
from keras.regularizers import l1_l2

dl = Preprocessor(balance=False)
x_train, y_train, x_test, y_test = dl.divided_data(ratio=0.5)
x_train, y_train =  dl.balance_raw_data(dl, x_train, y_train)
y_train = dl.transformed_labels(y_train)
y_test = dl.transformed_labels(y_test)
use_tensorboard = False # raskas, käytä vain tarvittaessa

opt = Adam(lr=1e-4)

model = Sequential()
model.add(normalization.BatchNormalization())
model.add(Dense(400, 
                activation='sigmoid', 
                input_dim=264,
                kernel_regularizer=l1_l2(1e-4, 1e-4)))
model.add(Dropout(rate=0.5))
model.add(Dense(300, 
                activation='sigmoid'))
model.add(Dropout(rate=0.2))
model.add(Dense(100, 
                activation='sigmoid',
                kernel_regularizer=l1_l2(1e-5, 1e-5)))
model.add(Dropout(rate=0.2))
model.add(Dense(70, activation='sigmoid'))
model.add(Dropout(rate=0.2))
model.add(Dense(10, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# toisella komentorivillä samassa hakemistossa: tensorboard --logdir ./Graph
tensorboardCB = keras.callbacks.TensorBoard(log_dir='./Graph', 
                                          histogram_freq=1,  
                                          write_graph=True, 
                                          write_images=True)

if use_tensorboard:
    model.fit(x_train, y_train,
            epochs=500,
            batch_size=8192,
            validation_data=(x_test, y_test),
            callbacks=[tensorboardCB])
else:
    model.fit(x_train, y_train,
            epochs=5000,
            batch_size=256,
            validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, batch_size=128)
print("Score: ", score)