import pandas as pd
import numpy as np

import pickle 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization, Activation
from tensorflow.keras import initializers 
from tensorflow.keras import regularizers 
from tensorflow.keras import constraints 

from sklearn.model_selection import train_test_split

from data_entry import data_entry
from ptn_io import isfileandnotempty, getfileswithname

import matplotlib.pyplot as plt

# Tryptophan (largest amino acid) = 0.67 nm in diameter 6.7 angstroms -> 7 A
# For 10 Tryptophan, 70 Angstroms x 70 Angstroms x 70 Angstroms
# Poole C and F J Owens, 'Introduction to Nanotechnology' Wiley 2003 p 315.
CUBIC_LENGTH_CONSTRAINT = 70

# Number of samples to be shoved into the network each round
BACTH_SIZE = 16

# Features per object
FEATURES_PER_GRID_POINT = 1

class cnn:
	def __init__(c, param = None):
		c.param = param


	def generate_model(c, input_shape):
		model = Sequential()
		model.add(Conv3D(3, 8, strides=(1, 1, 1), padding="same", input_shape=input_shape[1:]))
		model.add(BatchNormalization())
		model.add(Dense(8, activation='relu'))
		model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2))

		model.add(Conv3D(3, 16, strides=(1, 1, 1), padding="same", input_shape=input_shape[1:]))
		model.add(BatchNormalization())
		model.add(Dense(16, activation='relu'))
		model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2))

		model.add(Conv3D(3, 32, strides=(1, 1, 1), padding="same", input_shape=input_shape[1:]))
		model.add(BatchNormalization())
		model.add(Dense(32, activation='relu'))
		model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2))

		model.add(Dropout(0.2))
		model.add(Flatten())
		model.add(Dense(1))

		# Compiles the model
		model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse'])

		model.summary()

		return model

# Path name for storing all of the data
fdir = 'ptndata/'

# Load all of the obj file types and sort them by file name
files = getfileswithname(fdir, 'obj')
files.sort()

feature_set = []

# Load all of the objects into the feature set 
for file in files:
	filehandler = open(fdir + file, 'rb') 
	entry = pickle.load(filehandler)
	feature_set.append(np.reshape(entry.mat, (CUBIC_LENGTH_CONSTRAINT, CUBIC_LENGTH_CONSTRAINT, CUBIC_LENGTH_CONSTRAINT, 1)))

# Load energy scores from csv and sort them according to file name
energy_scores = pd.read_csv(fdir + 'energy.csv')
energy_scores.sort_values(by=['file'], inplace=True)

X = np.array(feature_set)
y = energy_scores['score'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

input_shape = (BACTH_SIZE, CUBIC_LENGTH_CONSTRAINT, CUBIC_LENGTH_CONSTRAINT, CUBIC_LENGTH_CONSTRAINT, FEATURES_PER_GRID_POINT)

if (True):
	cnn = cnn()
	model = cnn.generate_model(input_shape)

	history = model.fit(X_train, y_train, epochs = 100, batch_size = 16, verbose=1, validation_data=(X_test, y_test))

	data = pd.DataFrame({'abs_loss': [history.history['loss']], 'abs_val_loss': [history.history['val_loss']], 'rel_loss': [history.history['loss'] / np.mean(y_train)], 'rel_val_loss': [history.history['val_loss'] / np.mean(y_test)]})

	data.to_csv('data/data_1crnA0-10.csv')

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model absolute loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')

	savefig('cnn0_1crnA0-10_abs_loss.png')

	plt.plot(history.history['loss'] / np.mean(y_train))
	plt.plot(history.history['val_loss'] / np.mean(y_test))
	plt.title('model relative loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')

	savefig('cnn0_1crnA0-10_rel_loss.png')

	#plt.show()

# model.predict(X_train[0], verbose=0)