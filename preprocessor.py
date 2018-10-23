import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Preprocessor:

	def __init__(self, path='data/', balance=True, mutation_rate=5e-2, scale=True):
		self.balance 		 = balance
		self.scale 			 = scale
		self.mutation_rate 	 = mutation_rate
		self.data_path 		 = path
		self.raw_data_labels = pd.read_csv(self.data_path + 'train_labels.csv', header=None)
		self.unique_labels	 = np.unique(self.raw_data_labels)
		self.raw_data 		 = self.load_raw_data('train_data.csv')
		self.test_data		 = self.load_raw_data('test_data.csv')

	def load_raw_data(self, file):
		data = pd.read_csv(self.data_path + file, header=None).values
		if self.scale:
			ptp = data.ptp(0)
			for i in range(ptp.shape[0]):
				if ptp[i] == 0:
					ptp[i] = 0.5
			data = (data - data.min(0)) / ptp
		return data

	def balance_raw_data(self, data, labels, save_bal_data, bal_data_path):
		distribution = {}
		raw = np.hstack((labels, data))
		for label in self.unique_labels:
			distribution[int(label)] = 0
		for label in labels:
			distribution[int(label)] += 1
		distmax = distribution[max(distribution, key=distribution.get)]
		amount = 0
		for key in distribution:
			amount += distmax - distribution[key]
		count = 0
		tmp = np.empty((0, 265))
		for label in self.unique_labels:
			auxiliary_rows = raw[raw[:, 0] == label]
			for _ in range(distmax - distribution[label]):
				to_add = auxiliary_rows[np.random.randint(auxiliary_rows.shape[0])]
				for elem in range(to_add.shape[0] - 1):
					rnd = np.random.uniform(1 - self.mutation_rate, 1 + self.mutation_rate)
					to_add[elem + 1] *= rnd
				count += 1
				if count % 1000 == 0:
					print("Processed count: ", count, "/", amount)
				tmp = np.append(tmp, [to_add], axis=0)
		raw = np.vstack((raw, tmp))
		# optionally saves the data
		# boolean save_bal_data and save path is given to class method 'divided_data()'
		if save_bal_data:
			pd.DataFrame(raw).to_csv(bal_data_path)
		new_labels = raw[:, :1]
		new_features = raw[:, 1:]
		return new_features, new_labels

	def label_filter(self, row, label: int):
		#print(int(row[0])
		return row[0] == label

	# loads all feature vectors from training data and validation data for unsupervised learning
	# shape: (num_of_vectors, 264)
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
	def normalize_data(self, data):
		transformed = data
		print(type(data))
		return transformed

	# divides training data according to ratio for training purposes: (training_data, training_labels), (testing_data, testing_labels)
	# shape: (ratio*4263, col), ((1 - ratio)*4363, col)
	# loads preprocessed data from file or preprocesses the data and optionally saves to file
	def divided_data(self, ratio=0.5, bal_data_path=None, load_bal_data=True, save_bal_data=False):
		# path to balanced data file is used for both loading and saving the file
		if not bal_data_path:
			# appends ratio in percentages to file name
			bal_data_path = self.data_path + 'bal_data_ratio_' + str(int(100*ratio)) + '.csv'

		raw = np.hstack((self.raw_data_labels, self.raw_data))
		np.random.shuffle(raw)
		num = int(ratio * raw.shape[0])
		training_data = raw[:num, 1:]
		training_labels = raw[:num, :1]

		# balance the data
		if (self.balance and not load_bal_data):
			training_data, training_labels = self.balance_raw_data(training_data, training_labels, save_bal_data, bal_data_path)

		# use old balanced data file
		if (self.balance and load_bal_data):
			raw = pd.read_csv(bal_data_path, header=None).values
			# pandas puts headers as first column and row
			training_labels = raw[1:, 1:2]
			training_data = raw[1:, 2:]

		testing_data = raw[num:, 1:]
		testing_labels = raw[num:, :1]
		return training_data, training_labels, testing_data, testing_labels

	# returns columns (1-168)
	# shape: (row, 168)
	def get_rhythm_patterns(self, data):
		return data[:,0:168]

	# return columns (169-216)
	# shape: (row, 48)
	def get_chroma(self, data):
		return data[:,168:216]

	# return columns 217-264
	# shape: (row, 48)
	def get_mfcc(self, data):
		return data[:,216:]

	def save_result(self, prediction):
		pd.DataFrame(prediction).to_csv(self.data_path + 'result.csv')
