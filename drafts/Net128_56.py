import keras
from keras.layers import Input, LSTM, concatenate, Dense, Dropout, BatchNormalization, GRU, GaussianNoise
from preprocessor import Preprocessor
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.regularizers import l1_l2
import numpy as np

dl = Preprocessor(balance=True, scale=True, mutation_rate=0.15)
x_train, y_train, x_test, y_test = dl.divided_data(ratio=0.8, load_bal_data=False)
y_train = dl.transformed_labels(y_train)
y_test = dl.transformed_labels(y_test)

rhythm_input = Input(shape=(7, 24)) # timesteps, timestep dimension
chroma_input = Input(shape=(4, 12)) # timesteps, timestep dimension
mfcc_input 	 = Input(shape=(4, 12)) # timesteps, timestep dimension

rhythm_lstm  = LSTM(48, activation='tanh', kernel_regularizer=l1_l2(1e-6, 1e-7))(rhythm_input)
rhythm_lstm  = Dropout(rate=0.25)(rhythm_lstm)
rhythm_lstm  = Dense(48, activation='tanh', kernel_regularizer=l1_l2(1e-6, 1e-5))(rhythm_lstm)
rhythm_lstm  = Dropout(rate=0.5)(rhythm_lstm)

chroma_lstm  = LSTM(24, activation='tanh', kernel_regularizer=l1_l2(1e-6, 1e-7))(chroma_input)
chroma_lstm  = Dropout(rate=0.25)(chroma_lstm)
chroma_lstm  = Dense(24, activation='tanh', kernel_regularizer=l1_l2(1e-6, 1e-5))(chroma_lstm)
chroma_lstm  = Dropout(rate=0.5)(chroma_lstm)

mfcc_lstm 	 = LSTM(24, activation='tanh', kernel_regularizer=l1_l2(1e-6, 1e-7))(mfcc_input)
mfcc_lstm    = Dropout(rate=0.25)(mfcc_lstm)
mfcc_lstm  	 = Dense(24, activation='tanh', kernel_regularizer=l1_l2(1e-6, 1e-5))(mfcc_lstm)
mfcc_lstm    = Dropout(rate=0.5)(mfcc_lstm)

classifier 	 = concatenate([rhythm_lstm, chroma_lstm, mfcc_lstm], axis=-1)

for rate in range(3):
	classifier = Dense(72, activation='tanh', kernel_regularizer=l1_l2(1e-6, 1e-5))(classifier)
	classifier = Dropout(rate=0.5)(classifier)

classifier = Dense(10, activation='sigmoid', kernel_regularizer=l1_l2(1e-7, 1e-6))(classifier)

input_data = [
	np.reshape(x_train[:, :168], (x_train.shape[0], 7, 24)),
	np.reshape(x_train[:, 168:216], (x_train.shape[0], 4, 12)),
	np.reshape(x_train[:, 216:], (x_train.shape[0], 4, 12))
]

validation_data = [
	np.reshape(x_test[:, :168], (x_test.shape[0], 7, 24)),
	np.reshape(x_test[:, 168:216], (x_test.shape[0], 4, 12)),
	np.reshape(x_test[:, 216:], (x_test.shape[0], 4, 12))
]

tensorboardCB = keras.callbacks.TensorBoard(log_dir='./Graph', 
                                          histogram_freq=1,  
                                          write_graph=True, 
                                          write_images=True)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
sgd = SGD(momentum=0.1, nesterov=True)

model = Model(inputs=[rhythm_input, chroma_input, mfcc_input], outputs=classifier)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(input_data, y_train, 
				validation_data=(validation_data, y_test), 
				batch_size=256, 
				epochs=200)