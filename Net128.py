import keras
from keras.layers import Input, LSTM, concatenate, Dense, Dropout, BatchNormalization, GRU
from preprocessor import Preprocessor
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.regularizers import l1_l2
import numpy as np

dl = Preprocessor(balance=True, scale=True)
x_train, y_train, x_test, y_test = dl.divided_data(ratio=0.5)
y_train = dl.transformed_labels(y_train)
y_test = dl.transformed_labels(y_test)

rhythm_input = Input(shape=(7, 24)) # timesteps, timestep dimension
chroma_input = Input(shape=(4, 12)) # timesteps, timestep dimension
mfcc_input 	 = Input(shape=(4, 12)) # timesteps, timestep dimension

rhythm_lstm  = LSTM(48)(rhythm_input)
rhythm_lstm  = Dropout(rate=0.6)(rhythm_lstm)
rhythm_lstm  = Dense(10, activation='softmax')(rhythm_lstm)

chroma_lstm  = LSTM(24)(chroma_input)
chroma_lstm  = Dropout(rate=0.6)(chroma_lstm)
chroma_lstm  = Dense(10, activation='sigmoid')(chroma_lstm)

mfcc_lstm 	 = LSTM(24)(mfcc_input)
mfcc_lstm    = Dropout(rate=0.6)(mfcc_lstm)
mfcc_lstm  	 = Dense(10, activation='sigmoid')(mfcc_lstm)

classifier 	 = concatenate([rhythm_lstm, chroma_lstm, mfcc_lstm], axis=-1)

for rate in range(2):
	classifier = Dense(40, activation='sigmoid')(classifier)
	classifier = Dropout(rate=0.35)(classifier)

classifier = Dense(10, activation='sigmoid')(classifier)

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

adam = Adam(lr=1e-1, amsgrad=True)
sgd = SGD(momentum=0.1, nesterov=True)

model = Model(inputs=[rhythm_input, chroma_input, mfcc_input], outputs=classifier)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(input_data, y_train, validation_data=(validation_data, y_test), batch_size=256, epochs=1000)