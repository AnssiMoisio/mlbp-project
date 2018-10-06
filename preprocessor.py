import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Preprocessor:

	def __init__(self, path='data/', scale=True):
		self.scale = scale
		self.data_path = path
		self.raw_data = self.load_raw_data()
		self.raw_data_labels = pd.read_csv(self.data_path + 'train_labels.csv', header=None)

	def load_raw_data(self, scale=True):
		data = pd.read_csv(self.data_path + 'train_data.csv', header=None).values
		if self.scale:
			data = (data - data.min(0)) / data.ptp(0)
		return data

	# loads all feature vectors from training data and validation data for unsupervised learning
	# shape: (10907, 264)
	def all_feature_vectors(self):
		pass

	# transforms the label data to vectors, for example: 2 => (0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
	# shape: 
	def transformed_labels(self, data):
		unique_labels = np.unique(self.raw_data_labels)
		labels = np.zeros((data.shape[0], unique_labels.shape[0]))
		for i in range(data.shape[0]):
			labels[i][int(data[i]) - 1] = 1
		assert labels.shape == (data.shape[0], unique_labels.shape[0])
		return labels

	# normalises columns to [0.0, 1.0]
	def normalise_data(self):
		pass

	# divides training data according to ratio for training purposes: (training_data, training_labels), (testing_data, testing_labels)
	# shape: (ratio*4263, col), ((1 - ratio)*4363, col)
	def divided_data(self, ratio=0.5, normalise=False):
		raw = np.column_stack((self.raw_data_labels, self.raw_data))
		np.random.shuffle(raw)
		assert raw.shape == (4363, 265)
		num = int(ratio * raw.shape[0])
		training_data = raw[:num, 1:]
		training_labels = raw[:num, :1]
		testing_data = raw[num:, 1:]
		testing_labels = raw[num:, :1]
		return training_data, training_labels, testing_data, testing_labels

	# returns columns (2-169)
	# shape: (row, 168)
	def get_rhythm_patterns(self, data):
		pass

	# return columns (170-217)
	# shape: (row, 48)
	def get_chroma(self):
		pass

	# return columns 218-264
	# shape: (row, 48)
	def get_mfcc(self):
		pass