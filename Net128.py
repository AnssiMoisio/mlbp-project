import keras
from keras.layers import Input, LSTM, concatenate, Dense, Dropout, BatchNormalization, GRU, GaussianNoise
from preprocessor import Preprocessor
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.regularizers import l1_l2
import numpy as np
from matplotlib import pyplot as plt
import time

dl = Preprocessor(balance=False, scale=True, mutation_rate=0.15)
x_train, y_train, x_test, y_test = dl.divided_data(ratio=0.8, load_bal_data=False)
y_train = dl.transformed_labels(y_train)
y_test = dl.transformed_labels(y_test)

rhythm_input = Input(shape=(168,)) # timesteps, timestep dimension
chroma_input = Input(shape=(48,)) # timesteps, timestep dimension
mfcc_input 	 = Input(shape=(48,)) # timesteps, timestep dimension

rhythm  = Dense(168, activation='tanh', kernel_regularizer=l1_l2(1e-6, 1e-6))(rhythm_input)
rhythm  = Dropout(rate=0.3)(rhythm)
rhythm  = Dense(42, activation='softmax', kernel_regularizer=l1_l2(1e-6, 2e-6))(rhythm)
rhythm  = Dropout(rate=0.4)(rhythm)

chroma  = Dense(48, activation='tanh', kernel_regularizer=l1_l2(1e-6, 1e-6))(chroma_input)
chroma  = Dropout(rate=0.3)(chroma)
chroma  = Dense(40, activation='tanh', kernel_regularizer=l1_l2(1e-6, 1e-6))(chroma)
chroma  = Dropout(rate=0.4)(chroma)

mfcc  	 = Dense(48, activation='tanh', kernel_regularizer=l1_l2(1e-6, 1e-6))(mfcc_input)
mfcc    = Dropout(rate=0.3)(mfcc)
mfcc  	 = Dense(40, activation='tanh', kernel_regularizer=l1_l2(1e-6, 1e-6))(mfcc)
mfcc    = Dropout(rate=0.4)(mfcc)

classifier 	 = concatenate([rhythm, chroma, mfcc], axis=-1)

for rate in range(3):
	classifier = Dense(120, activation='tanh', kernel_regularizer=l1_l2(1e-6, 1e-6))(classifier)
	classifier = Dropout(rate=0.35)(classifier)

classifier = Dense(10, activation='softmax', kernel_regularizer=l1_l2(1e-7, 1e-6))(classifier)

input_data = [
	x_train[:, :168],
	x_train[:, 168:216],
	x_train[:, 216:]
]

validation_data = [
	x_test[:, :168],
	x_test[:, 168:216],
	x_test[:, 216:]
]

tensorboardCB = keras.callbacks.TensorBoard(log_dir='./Graph', 
                                          histogram_freq=1,  
                                          write_graph=True, 
                                          write_images=True)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
sgd = SGD(momentum=0.1, nesterov=True)

model = Model(inputs=[rhythm_input, chroma_input, mfcc_input], outputs=classifier)
model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])

start_time = time.time()
model.fit(input_data, y_train, 
				validation_data=(validation_data, y_test), 
				batch_size=256, 
				epochs=60000)
print('Measured runtime: %.2f seconds.' % (time.time() - start_time))


data = dl.test_data
test_data = [
	data[:, :168],
	data[:, 168:216],
	data[:, 216:]
]
prediction = model.predict(test_data)
y_classes = prediction.argmax(axis=-1) # convert probabilities into labels

plt.figure(2, figsize=(10, 8))
plt.title("prediction label distribution")
plt.hist(y_classes, range=(-0.5,9.5), bins=10, ec='black')
plt.show()

y_classes = np.subtract(y_classes, [-1]*6544)
dl.save_result(y_classes)