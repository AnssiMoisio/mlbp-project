import pandas as pd
import numpy as np

class Preprocessor:

	def __init__(self, path='data/'):
		self.data_path = path
		self.raw_data = pd.read_csv(self.data_path + 'train_data.csv', header=None)
		self.raw_data_labels = pd.read_csv(self.data_path + 'train_labels.csv', header=None)

	# loads all feature vectors from training data and validation data for unsupervised learning
	# shape: (10907, 264)
	def all_feature_vectors(self):
		pass

	# transforms the label data to vectors, for example: 2 => (0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
	# shape: 
	def transformed_labels(self):
		pass

	# normalises columns to [0.0, 1.0]
	# shape: (4363, 264)
	def normalise_data(self):
		pass

	# divides training data according to ratio for training purposes: (training_data, training_labels), (testing_data, testing_labels)
	# shape: (ratio*4263, col), ((1 - ratio)*4363, col)
	def divided_data(self, ratio=0.5, normalise=False):
		raw = self.raw_data_labels.append(self.raw_data).sample(frac=1)
		assert raw.shape == (4363, 265)
		num = ratio * raw_data.shape[0]
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