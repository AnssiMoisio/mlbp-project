# CS-E3210 MLBP 2018 - Data analysis project

- preprocessor.py // for training
	- load_training_data() // all training data in shape (4363, 264)
	- normalise_data(data) // normalizes columns to [0.0, 1.0]
	- divide_data(data=load_training_data(), ratio=0.5) // divides data to two sets: (ratio * 4363, (1 - ratio) * 4363)
	- get_rhythm_patterns(original=False) // 
	- get_chroma(original=False) // 
	- get_mfcc(original=False) // 

- classifier.py
	- 

label = [1, 10]
label = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]