import pandas as pd
import numpy as np
from numpy import asarray
from time import time

from sklearn.preprocessing import OneHotEncoder

import pickle 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization, Activation, Reshape, Conv1D, MaxPooling1D
from tensorflow.keras import initializers 
from tensorflow.keras import regularizers 
from tensorflow.keras import constraints 

from sklearn.model_selection import train_test_split

from data_entry import data_entry
from ptn_io import isfileandnotempty, getfileswithname
from grid_point import grid_point

import matplotlib.pyplot as plt

import math
import random
# For HPC 
# qsubi -pe smp 4 -l m_mem_free=5G -l h_vmem=5G
# screen -S ptn
# module load python/3.6.8
# source venv/bin/activate
# pip3 install --user --upgrade tensorflow
# python3.6 -i cnn.py

# Tryptophan (largest amino acid) = 0.67 nm in diameter 6.7 angstroms -> 7 A
# For 10 Tryptophan, 70 Angstroms x 70 Angstroms x 70 Angstroms
# Poole C and F J Owens, 'Introduction to Nanotechnology' Wiley 2003 p 315.
CUBIC_LENGTH_CONSTRAINT = 70

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
def grid2atomtype(mat, atom_type, atom_type_encoder):
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
def grid2atom(mat, atom_pos, atom_pos_encoder):
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


# Given a matrix, return the minimum required dimensions in order to capture all non-zero values.
def find_bounds(mat):
	x = [i for i in range(CUBIC_LENGTH_CONSTRAINT) if (np.array(mat[i]) != 0.0).any()]
	x_min = min(x)
	x_max = max(x)

	y = [i for i in range(CUBIC_LENGTH_CONSTRAINT) for j in range(x_min, x_max) if (np.array(mat[j][i]) != 0.0).any()]
	y_min = min(y)
	y_max = max(y)

	z = [i for i in range(CUBIC_LENGTH_CONSTRAINT) for j in range(x_min, x_max) for k in range(y_min, y_max) if (np.array(mat[j][k][i]) != 0.0).any()]
	z_min = min(z)
	z_max = max(z)

	return x_min, y_min, z_min, x_max, y_max, z_max


# Given new bounds and old bounds, return the proper updated bounds.
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


# This cnn class stores all necessary functions of the cnn networks under investigation.
class cnn:
	def __init__(c, param = None):
		c.param = param

	# Generate 3D CNN model for rosetta or mse--basically for any single value output.
	def generate_model_rosetta_mse(c, input_shape):
		model = Sequential()
		model.add(Conv3D(filters=3, kernel_size=8, strides=(1, 1, 1), padding="same", input_shape=input_shape[1:]))
		model.add(BatchNormalization())
		model.add(Dense(8, activation='relu'))
		model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2))
		model.add(Conv3D(filters=3, kernel_size=16, strides=(1, 1, 1), padding="same", input_shape=input_shape[1:]))
		model.add(BatchNormalization())
		model.add(Dense(16, activation='relu'))
		model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2))
		model.add(Conv3D(filters=3, kernel_size=32, strides=(1, 1, 1), padding="same", input_shape=input_shape[1:]))
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


	# Generate 3D CNN model for contact map--for square output.
	def generate_model_contact_map_3d(c, input_shape, output_shape):
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


	# Generate 1D CNN model for contact map.
	def generate_model_contact_map_1d(c, input_shape, output_shape):
		model = Sequential()
		model.add(Conv1D(filters=6, kernel_size=16, strides=(1), padding="same", input_shape=input_shape[1:]))
		model.add(BatchNormalization())
		model.add(Dense(16, activation='relu'))
		model.add(MaxPooling1D(pool_size=(2), strides=2))
		model.add(Conv1D(filters=6, kernel_size=16, strides=(1), padding="same", input_shape=input_shape[1:]))
		model.add(BatchNormalization())
		model.add(Dense(32, activation='relu'))
		model.add(MaxPooling1D(pool_size=(2), strides=2))
		model.add(Conv1D(filters=6, kernel_size=16, strides=(1), padding="same", input_shape=input_shape[1:]))
		model.add(BatchNormalization())
		model.add(Dense(64, activation='relu'))
		model.add(MaxPooling1D(pool_size=(2), strides=2))
		model.add(Dropout(0.2))
		model.add(Flatten())
		model.add(Dense(output_shape[2] * output_shape[2]))
		model.add(Reshape(output_shape[1:]))
		model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse'])
		model.summary()
		return model


# Given a set of files storing entry objects and their directory location, return their feature dimensions such as the positional atom types and the bounds for the matrix.
def load_feature_dimensions(files, fdir = 'ptndata_10H/'):
	x_min, y_min, z_min, x_max, y_max, z_max = CUBIC_LENGTH_CONSTRAINT, CUBIC_LENGTH_CONSTRAINT, CUBIC_LENGTH_CONSTRAINT, 0, 0, 0
	atom_pos = []
	for i, file in enumerate(files):
		print('Percentage complete: ', round(i / len(files) * 100, 2), '%', sep='')
		entry = pickle.load(open(fdir + file, 'rb'))
		new_x_min, new_y_min, new_z_min, new_x_max, new_y_max, new_z_max = find_bounds(grid2logical(entry.mat))
		x_min, y_min, z_min, x_max, y_max, z_max = update_bounds(new_x_min, new_y_min, new_z_min, new_x_max, new_y_max, new_z_max, x_min, y_min, z_min, x_max, y_max, z_max)
		#print(f'x: [{x_min},{x_max}]\ty: [{y_min},{y_max}]\tx: [{z_min},{z_max}]')
		atom_pos = get_all_atoms(entry.mat, atom_pos)
	atom_pos.append('None')

	return atom_pos, x_min, y_min, z_min, x_max, y_max, z_max


# This is a generator function for files containing entry objects in the given location. These objects, due to their large size, are fed into the CNN one at a time as a memory optimization step.
def sample_gen(files, feature_set, atom_type, atom_type_encoder, atom_pos, atom_pos_encoder, energy_scores, x_min, y_min, z_min, x_max, y_max, z_max, fdir='ptndata_10H/'):
	for q, file in enumerate(files):
		entry = pickle.load(open(fdir + file, 'rb'))
		a = grid2logical(entry.mat)
		b = grid2atomtype(entry.mat, atom_type, atom_type_encoder)
		c = grid2atom(entry.mat, atom_pos, atom_pos_encoder)
		dm_output = entry.dm
		# rosetta_score, mse_score
		#y = dm_output
		#y = np.reshape(y, (len(y), len(y[0][0]), len(y[0][0])))
		#y = y.astype(float)
		y = energy_scores.loc['ptndata_10H/' + file]['rosetta_score']
		for i in range(len(feature_set[0])):
			for j in range(len(feature_set[0][0])):
				for k in range(len(feature_set[0][0][0])):
					feature_set[0][i][j][k] = [a[x_min + i][y_min + j][z_min + k]] + b[x_min + i][y_min + j][z_min + k].tolist() + c[x_min + i][y_min + j][z_min + k].tolist()
		y = np.array(y)
		y = y.reshape(-1,1)	
		yield (feature_set, y)


# This is almost like sample_gen, except it is a function instead of a generator function. This is used for generating the validation data before training the CNN. It generates the validation samples for all three of the metrics.
def sample_loader(files, feature_set_, atom_type, atom_type_encoder, atom_pos, atom_pos_encoder, energy_scores, x_min, y_min, z_min, x_max, y_max, z_max, fdir='ptndata_10H/'):
	# Number of atoms is set to a hard cut off so the convolution network has a constant size 
	NUMBER_OF_ATOMS = 10 

	y_rosetta = []
	y_mse = []
	y_dm = []
	for q, file in enumerate(files):
		print('Percentage complete: ', round(q / len(files) * 100, 2), '%', sep='')
		entry = pickle.load(open(fdir + file, 'rb'))
		a = grid2logical(entry.mat)
		b = grid2atomtype(entry.mat, atom_type, atom_type_encoder)
		c = grid2atom(entry.mat, atom_pos, atom_pos_encoder)

		dm_output = select_region_dm(entry.dm, (NUMBER_OF_ATOMS, NUMBER_OF_ATOMS))

		#y = np.reshape(y, (len(y), len(y[0][0]), len(y[0][0])))
		#y = y.astype(float)
		y_rosetta.append(energy_scores.loc['ptndata_10H/' + file]['rosetta_score'])
		y_mse.append(energy_scores.loc['ptndata_10H/' + file]['mse_score'])
		y_dm.append(dm_output)
		for i in range(len(feature_set_[0])):
			for j in range(len(feature_set_[0][0])):
				for k in range(len(feature_set_[0][0][0])):
					feature_set_[q][i][j][k] = [a[x_min + i][y_min + j][z_min + k]] + b[x_min + i][y_min + j][z_min + k].tolist() + c[x_min + i][y_min + j][z_min + k].tolist()

	y_rosetta = np.array(y_rosetta)
	y_rosetta = y_rosetta.reshape(-1,1)		

	y_mse = np.array(y_mse)
	y_mse = y_mse.reshape(-1,1)	

	y_dm = np.reshape(y_dm, (len(y_dm), len(y_dm[0]), len(y_dm[0][0])))
	y_dm = y_dm.astype(float)

	return feature_set_, y_rosetta, y_mse, y_dm


def select_region_dm(dm, shape):
	return np.array([[[dm[0][i][j]] for i in range(shape[0])] for j in range(shape[1])])


# Given the location of a directory with entry objects storing data for the 1D CNN, return the necessary features and target values for the network.
def conv1d_primary_seq_dm(fdir='ptndata_1dconv/'):
	start_time = time()

	files = getfileswithname(fdir, 'obj')
	#random.shuffle(files)

	# Number of atoms is set to a hard cut off so the convolution network has a constant size 
	NUMBER_OF_ATOMS = 10

	total_samples = len(files)
	validation_split = 0.2

	training_samples = int(total_samples * (1 - validation_split))
	validation_samples = int(total_samples * validation_split)

	feature_set = np.array([ [ [0] for _ in range(20 * NUMBER_OF_ATOMS) ] for _ in range(total_samples) ])
	y = []
	for i, file in enumerate(files):
		entry = pickle.load(open(fdir + file, 'rb'))

		dm_output = select_region_dm(entry.dm, (NUMBER_OF_ATOMS, NUMBER_OF_ATOMS))
		y.append(dm_output)

		ordinal_features = entry.ordinal_features
		one_hot_features = entry.one_hot_features

		sample_atom_list = []

		if len(one_hot_features) < NUMBER_OF_ATOMS:
			continue

		for j in range(NUMBER_OF_ATOMS):

			row = one_hot_features[j]
			if type(row) == list:
				sample_atom_list += row
			else:
				sample_atom_list += row.tolist()
		feature_set[i] = np.array(sample_atom_list).reshape(-1, 1)

	feature_set = np.array(feature_set)
	input_shape = feature_set.shape

	y = np.reshape(y, (len(y), len(y[0]), len(y[0][0])))
	y = y.astype(float)
	output_shape = y.shape

	model = cnn.generate_model_contact_map_1d(input_shape, output_shape)
	history = model.fit(feature_set, y, batch_size=10, epochs=10, verbose=1, validation_split=0.2)

	return model, history, feature_set, y


def conv3d_tertiary_seq_rosetta_mse_dm(fdir='ptndata_10H/'):

	start_time = time()
	total_samples = 1000
	validation_split = 0.2

	training_samples = int(total_samples * (1 - validation_split))
	validation_samples = int(total_samples * validation_split)

	# Path name for storing all of the data
	#fdir = 'ptndata_small/'
	#fdir = '/Users/ethanmoyer/Projects/data/ptn/ptndata_10H/'
	print('Loading files...')
	# Load all of the obj file types and sort them by file name
	files = getfileswithname(fdir, 'obj')
	files.sort()

	files = files[:total_samples]

	energy_scores = pd.read_csv(fdir + 'energy_local_dir.csv', index_col='file')

	files = [file for file in files if 'ptndata_10H/' + file in energy_scores.index]

	training_files = files[:training_samples]
	validation_files = files[training_samples:]

	# Index for the four main types of atoms and None that will indexed when looping through each entry
	atom_type = ['C', 'N', 'O', 'S', 'None']
	atom_type_data = pd.Series(atom_type)
	atom_type_encoder = np.array(pd.get_dummies(atom_type_data))

	print('Detemining positional atom types and smallest window size of the data ...')
	# Loop through each file and make a list of all of the atoms present.

	atom_pos, x_min, y_min, z_min, x_max, y_max, z_max = load_feature_dimensions(files, fdir)

	# Format the position specific atom list so it can be used as one-hot encoding in the network
	atom_pos_data = pd.Series(atom_pos)
	atom_pos_encoder = np.array(pd.get_dummies(atom_pos_data))

	# Initialize the feature set
	feature_set = np.array([[[[ [0] * (1 + len(atom_type) + len(atom_pos)) for i in range(x_min, x_max)] for j in range(y_min, y_max)] for k in range(z_min, z_max)] for q in range(1)])

	feature_set_ = np.array([[[[ [0] * (1 + len(atom_type) + len(atom_pos)) for i in range(x_min, x_max)] for j in range(y_min, y_max)] for k in range(z_min, z_max)] for q in range(validation_samples)])

	# Define input and output shape
	input_shape = feature_set.shape
	#output_shape = y.shape

	#cnn = cnn()
	model = cnn.generate_model_rosetta_mse(input_shape)
	#model = cnn.generate_model_contact_map(input_shape, output_shape)

	print('Generating validation data ...')
	# Load all of the objects into the feature set 
	feature_set, y_rosetta, y_mse, y_dm = sample_loader(validation_files, feature_set_, atom_type, atom_type_encoder, atom_pos, atom_pos_encoder, energy_scores, x_min, y_min, z_min, x_max, y_max, z_max, fdir)

	print('Running model on training data...')
	history = model.fit(sample_gen(training_files, feature_set, atom_type, atom_type_encoder, atom_pos, atom_pos_encoder, energy_scores, x_min, y_min, z_min, x_max, y_max, z_max, fdir), steps_per_epoch=1,epochs = 200, verbose=1, use_multiprocessing=True, validation_data=(feature_set, y_rosetta)) #, 
	print('Time elapsed:', time() - start_time)

	data = pd.DataFrame({'abs_loss': [history.history['loss']], 'abs_val_loss': [history.history['val_loss']]})
	data.to_csv('figures/1crnAH10_ros.csv')
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model absolute loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('figures/1crnAH10_ros_abs_loss.png')
	plt.clf()

cnn = cnn()
conv3d_tertiary_seq_rosetta_mse_dm()
