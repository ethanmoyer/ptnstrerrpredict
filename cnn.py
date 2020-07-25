import pandas as pd
import numpy as np
from numpy import asarray
from time import time

from sklearn.preprocessing import OneHotEncoder

import pickle 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization, Activation, Reshape
from tensorflow.keras import initializers 
from tensorflow.keras import regularizers 
from tensorflow.keras import constraints 

from sklearn.model_selection import train_test_split

from data_entry import data_entry
from ptn_io import isfileandnotempty, getfileswithname
from grid_point import grid_point

import matplotlib.pyplot as plt

import math
# size test
# getsizeof(feature_set)*getsizeof(feature_set[0])*getsizeof(feature_set[0][0])*getsizeof(feature_set[0][0][0])*getsizeof(feature_set[0][0][0][0][0])
# 75698798592000

# Tryptophan (largest amino acid) = 0.67 nm in diameter 6.7 angstroms -> 7 A
# For 10 Tryptophan, 70 Angstroms x 70 Angstroms x 70 Angstroms
# Poole C and F J Owens, 'Introduction to Nanotechnology' Wiley 2003 p 315.
CUBIC_LENGTH_CONSTRAINT = 70

# Number of samples to be shoved into the network each round
BACTH_SIZE = 16

# Given an object loaded matrix of grid points, return a logical matrix representing atomic positions
def grid2logical(mat):
	a = len(mat)
	mat_ = [[[ [] for _ in range(a)] for _ in range(a)] for _ in range(a)]
	for i in range(len(mat)):
		for j in range(len(mat[0])):
			for k in range(len(mat[0][0])):
				mat_[i][j][k] = mat[i][j][k].occupancy
	return mat_


# Given an object loaded matrix of grid points, return a matrix of atom types into general categories {'N', 'O', 'C', 'S'}
def grid2atomtype(mat):
	a = len(mat)
	mat_ = [[[ [] for _ in range(a)] for _ in range(a)] for _ in range(a)]

	for i in range(len(mat)):
		for j in range(len(mat[0])):
			for k in range(len(mat[0][0])):
				atom = mat[i][j][k].atom
				if atom is None:
					mat_[i][j][k] = atom_type_encoder[atom_type.index("None")]
				else:
					mat_[i][j][k] = atom_type_encoder[atom_type.index(atom[:1])]
	return mat_


# Given an object loaded matrix of grid points, return a matrix of specific atom types
def grid2atom(mat):
	a = len(mat)
	mat_ = [[[ [] for _ in range(a)] for _ in range(a)] for _ in range(a)]

	for i in range(len(mat)):
		for j in range(len(mat[0])):
			for k in range(len(mat[0][0])):
				atom = mat[i][j][k].atom
				if atom is None:
					mat_[i][j][k] = atom_pos_encoder[atom_pos.index("None")]
				else:
					mat_[i][j][k] = atom_pos_encoder[atom_pos.index(atom)]
	return mat_


# Given an object loaded matrix of grid points, return a list of unique atoms.
def get_all_atoms(mat, atoms):
	for i in range(len(mat)):
		for j in range(len(mat[0])):
			for k in range(len(mat[0][0])):
				atom = mat[i][j][k].atom
				if atom is not None:
					atoms.append(atom)
	return list(set(atoms))


def find_bounds(mat):
	x = [i for i in range(CUBIC_LENGTH_CONSTRAINT) if (np.array(mat[i]) != 0.0).any()]
	x_min = min(x)
	x_max = max(x)

	y = [i for i in range(CUBIC_LENGTH_CONSTRAINT) if (np.array(mat[x_min][i]) != 0.0).any()]
	y_min = min(y)
	y_max = max(y)

	z = [i for i in range(CUBIC_LENGTH_CONSTRAINT) if (np.array(mat[x_min][y_min][i]) != 0.0).any()]
	z_min = min(z)
	z_max = max(z)

	return x_min, y_min, z_min, x_max, y_max, z_max


def update_bounds(new_x_min, new_y_min, new_z_min, new_x_max, new_y_max, new_z_max, x_min, y_min, z_min, x_max, y_max, z_max):
	if new_x_min < x_min:
		x_min = new_x_min

	if new_y_min < y_min:
		y_min = new_y_min

	if new_z_min < z_min:
		z_min = new_z_min

	if new_x_max > x_max:
		x_max = new_x_max

	if new_y_max > y_max:
		y_max = new_y_max

	if new_z_max > z_max:
		z_max = new_z_max	

	return x_min, y_min, z_min, x_max, y_max, z_max


def load_data(file_handler, block_size=10000):
    block = []
    for line in file_handler:
        block.append(line)
        if len(block) == block_size:
            yield block
            block = []

    # don't forget to yield the last block
    if block:
        yield block


class cnn:
	def __init__(c, param = None):
		c.param = param

	# Generate the CNN model
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

		model.add(Dropout(0.5))
		model.add(Flatten())
		model.add(Dense(1))

		# Compiles the model
		model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse'])

		model.summary()

		return model

	def generate_model_contact_map(c, input_shape, output_shape):
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

		model.add(Dropout(0.5))
		model.add(Flatten())
		model.add(Dense(output_shape[2] * output_shape[2]))
		model.add(Reshape(output_shape[1:]))

		model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse'])

		model.summary()

		return model

start_time = time()

samples = 5

# Path name for storing all of the data
fdir = 'ptndata_10H/'
#fdir = '/Users/ethanmoyer/Projects/data/ptn/ptndata_10H/'
print('Loading files...')
# Load all of the obj file types and sort them by file name
files = getfileswithname(fdir, 'obj')
files.sort()

# Index for the four main types of atoms and None that will indexed when looping through each entry
atom_type = ['C', 'N', 'O', 'S', 'None']
atom_type_data = pd.Series(atom_type)
atom_type_encoder = np.array(pd.get_dummies(atom_type_data))

# Loading files
energy_scores = pd.read_csv(fdir + 'energy_local_dir.csv')
energy_scores.sort_values(by=['file'], inplace=True)

# Initialize a list of enzymes
atom_pos = []

dm_output = []
i = 0
print('Loading positional atom types, distance matrix, and required size of window ...')
# Loop through each file and make a list of all of the atoms present.

x_min, y_min, z_min, x_max, y_max, z_max = CUBIC_LENGTH_CONSTRAINT, CUBIC_LENGTH_CONSTRAINT, CUBIC_LENGTH_CONSTRAINT, 0, 0, 0
for file in files[:samples]:
	print('File complete:', i / len(files) * 100)

	if all([file not in energy_file for energy_file in energy_scores['file']]):
		continue

	i += 1
	filehandler = open(fdir + file, 'rb') 
	entry = pickle.load(filehandler)
	new_x_min, new_y_min, new_z_min, new_x_max, new_y_max, new_z_max = find_bounds(grid2logical(entry.mat))
	x_min, y_min, z_min, x_max, y_max, z_max = update_bounds(new_x_min, new_y_min, new_z_min, new_x_max, new_y_max, new_z_max, x_min, y_min, z_min, x_max, y_max, z_max)

	atom_pos = get_all_atoms(entry.mat, atom_pos)
	dm_output.append(entry.dm)

# Format the position specific atom list so it can be used as one-hot encoding in the network
atom_pos.append('None')
atom_pos_data = pd.Series(atom_pos)
atom_pos_encoder = np.array(pd.get_dummies(atom_pos_data))

# Initialize the feature set
feature_set = np.array([[[[ [0] * (1 + len(atom_type) + len(atom_pos)) for i in range(x_min, x_max)] for j in range(y_min, y_max)] for k in range(z_min, z_max)] for q in range(samples)])

#feature_set = []

if True:
	print('Loading main features...')
	q = 0
	# Load all of the objects into the feature set 
	for file in files[:samples]:
		print('File complete:' , q / len(files) * 100)
		
		if all([file not in energy_file for energy_file in energy_scores['file']]):
			continue

		filehandler = open(fdir + file, 'rb') 
		entry = pickle.load(filehandler)
		a = grid2logical(entry.mat)
		b = grid2atomtype(entry.mat)
		c = grid2atom(entry.mat)

		# Append all of the feature categories into dimension
		#sample = [[[ [a[i][j][k]] + b[i][j][k].tolist() + c[i][j][k].tolist() for i in range(x_min, x_max)] for j in range(y_min, y_max)] for k in range(z_min, z_max)]

		# Append each sample to the feature set
		for i in range(len(feature_set)):
			for j in range(len(feature_set[0])):
				for k in range(len(feature_set[0][0])):
					feature_set[q][i][j][k] = [a[i][j][k]] + b[i][j][k].tolist() + c[i][j][k].tolist()
		i += q

if True:
	# Load energy scores from csv and sort them according to file name

	# Split features and outputs
	#X = np.array(feature_set)
	X = feature_set
	#use this later y = energy_scores['mse_score'].values # rosetta_score,mse_score
	y = energy_scores['rosetta_score'].values[:samples]

	#y = dm_output
	#y = np.reshape(y, (len(y), len(y[0][0]), len(y[0][0])))
	#y = y.astype(float)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
	print('Running model...')

	cnn = cnn()

	input_shape = (y.shape[0], x_max - x_min, y_max - y_min, z_max - z_min, X.shape[4])
	output_shape = y.shape
	model = cnn.generate_model(input_shape)

#model = cnn.generate_model_contact_map(input_shape, output_shape)
if True:
	history = model.fit(X_train, y_train, epochs = 10, batch_size = 10, verbose=1, validation_data=(X_test, y_test))
	print('Time elapsed:', time() - start_time)


if False:
	data = pd.DataFrame({'abs_loss': [history.history['loss']], 'abs_val_loss': [history.history['val_loss']], 'rel_loss': [history.history['loss'] / np.mean(y_train)], 'rel_val_loss': [history.history['val_loss'] / np.mean(y_test)]})
	data.to_csv('figures/1crnA5H_mse.csv')
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model absolute loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('figures/1crnA10H_ros_abs_loss.png')
	plt.clf()
	a = [math.sqrt(e) for e in history.history['loss']]
	plt.plot(a / np.mean(y_train))
	a = [math.sqrt(e) for e in history.history['val_loss']]
	plt.plot(a / np.mean(y_test))
	plt.title('model relative loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('figures/1crnA10H_ros_rel_loss.png')

	#plt.show()

# model.predict(X_train[0], verbose=0)