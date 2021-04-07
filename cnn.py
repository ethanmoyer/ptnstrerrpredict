import pandas as pd
import numpy as np
from numpy import asarray
from time import time

from sklearn.preprocessing import OneHotEncoder

import pickle 
import tempfile

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization, Activation, Reshape, Conv1D, MaxPooling1D
from tensorflow.keras import initializers 
from tensorflow.keras import regularizers 
from tensorflow.keras import constraints 
from tensorflow.keras.callbacks import EarlyStopping


from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

from data_entry import data_entry
from ptn_io import isfileandnotempty, getfileswithname
from grid_point import grid_point
from geo import geo_alignpoints, geo_distmat_to3d

import matplotlib.pyplot as plt

import math
import random
# For HPC 
# source venv/bin/activate
# qsubi -pe smp 4 -l m_mem_free=5G -l h_vmem=5G
# screen -S ptn
# module load python/3.6.8
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
		model.add(Dense(output_shape[0] * output_shape[1]))
		model.add(Reshape(output_shape))
		model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse'])
		model.summary()
		return model


	# Generate 1D CNN model for contact map.
	def generate_model_contact_map_1d(c, input_shape, output_shape):
		model = Sequential()
		model.add(Conv1D(filters=6, kernel_size=6, strides=(1), padding="same", input_shape=input_shape[1:]))
		model.add(BatchNormalization())
		model.add(Dense(6, activation='relu'))
		model.add(MaxPooling1D(pool_size=(2), strides=2))
		model.add(Conv1D(filters=6, kernel_size=6, strides=(1), padding="same", input_shape=input_shape[1:]))
		model.add(BatchNormalization())
		model.add(Dense(6, activation='relu'))
		model.add(MaxPooling1D(pool_size=(2), strides=2))
		model.add(Conv1D(filters=6, kernel_size=6, strides=(1), padding="same", input_shape=input_shape[1:]))
		model.add(BatchNormalization())
		model.add(Dense(6, activation='relu'))
		model.add(MaxPooling1D(pool_size=(2), strides=2))
		model.add(Dropout(0.2))
		model.add(Flatten())
		model.add(Dense(output_shape[0] * output_shape[1]))
		model.add(Reshape(output_shape))
		model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse'])
		model.summary()
		return model


	def generate_model_aan_contact_map_1d(c, input_shape, output_shape):
		model = Sequential()
		model.add(Dense(15, input_shape=input_shape[1:]))
		model.add(Flatten())
		model.add(Dense(output_shape[0] * output_shape[1]))
		model.add(Reshape(output_shape))
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

		y = dm_output[0].tolist()
		y = np.reshape(y, (1, len(y[0]), len(y[0])))
		y = y.astype(float)
		#y = energy_scores.loc['ptndata_10H/' + file]['mse_score']
		#y = np.array(y)
		#y = y.reshape(-1,1)	
		for i in range(len(feature_set[0])):
			for j in range(len(feature_set[0][0])):
				for k in range(len(feature_set[0][0][0])):
					feature_set[0][i][j][k] = [a[x_min + i][y_min + j][z_min + k]] + b[x_min + i][y_min + j][z_min + k].tolist() + c[x_min + i][y_min + j][z_min + k].tolist()

		yield (feature_set, y)


# This is almost like sample_gen, except it is a function instead of a generator function. This is used for generating the validation data before training the CNN. It generates the validation samples for all three of the metrics.
def sample_loader(files, feature_set_, atom_type, atom_type_encoder, atom_pos, atom_pos_encoder, energy_scores, x_min, y_min, z_min, x_max, y_max, z_max, fdir='ptndata_10H/'):
#if True:
	y_rosetta = []
	y_mse = []
	y_dm = []
	for q, file in enumerate(files):
		print('Percentage complete: ', round(q / len(files) * 100, 2), '%', sep='')
		entry = pickle.load(open(fdir + file, 'rb'))
		a = grid2logical(entry.mat)
		b = grid2atomtype(entry.mat, atom_type, atom_type_encoder)
		c = grid2atom(entry.mat, atom_pos, atom_pos_encoder)
#
		#y = np.reshape(y, (len(y), len(y[0][0]), len(y[0][0])))
		#y = y.astype(float)
		y_rosetta.append(energy_scores.loc['ptndata_10H/' + file]['rosetta_score'])
		y_mse.append(energy_scores.loc['ptndata_10H/' + file]['mse_score'])
		y_dm.append(entry.dm)
		for i in range(len(feature_set_[0])):
			for j in range(len(feature_set_[0][0])):
				for k in range(len(feature_set_[0][0][0])):
					feature_set_[q][i][j][k] = [a[x_min + i][y_min + j][z_min + k]] + b[x_min + i][y_min + j][z_min + k].tolist() + c[x_min + i][y_min + j][z_min + k].tolist()

	y_rosetta = np.array(y_rosetta)
	y_rosetta = y_rosetta.reshape(-1,1)		

	y_mse = np.array(y_mse)
	y_mse = y_mse.reshape(-1,1)	

	y_dm = np.reshape(y_dm, (len(y_dm), len(y_dm[0][0]), len(y_dm[0][0])))
	y_dm = y_dm.astype(float)

	return feature_set_, y_rosetta, y_mse, y_dm


def select_region_dm(dm, shape):
	return np.array([[ [dm[k][j][i] for i in range(shape[1])] for j in range(shape[0])] for k in range(len(dm))])


# Given the location of a directory with entry objects storing data for the 1D CNN, return the necessary features and target values for the network.
def conv1d_primary_seq_dm(fdir='ptndata_1dconv/'):
#if True:
	# Number of atoms is set to a hard cut off so the convolution network has a constant size 
	NUMBER_OF_AA = 11
	NUMBER_OF_AA2 = NUMBER_OF_AA
	#
	start_time = time()
	fdir = '/Users/ethanmoyer/Projects/data/ptn/ptndata_1dconv/'
	files = getfileswithname(fdir, 'obj')
	files = [file for file in files if pickle.load(open(fdir + file, 'rb')).dm.shape == (NUMBER_OF_AA2, NUMBER_OF_AA2) and len(pickle.load(open(fdir + file, 'rb')).one_hot_features) == NUMBER_OF_AA]
	#random.shuffle(files)
#
	total_samples = len(files)
	files = files[:total_samples]
	validation_split = 0.2
#
#
	training_samples = int(total_samples * (1 - validation_split))
	validation_samples = int(total_samples * validation_split)
#
	feature_set = np.array([ [ [0] for _ in range(20 * NUMBER_OF_AA) ] for _ in range(total_samples) ])
	y = []
#
	for i, file in enumerate(files):
#
		entry = pickle.load(open(fdir + file, 'rb'))
#
		one_hot_features = entry.one_hot_features
#
		sample_atom_list = []
#
		y.append(entry.dm)
		for j in range(NUMBER_OF_AA):
#
			row = one_hot_features[j]
			if type(row) == list:
				sample_atom_list += row
			else:
				sample_atom_list += row.tolist()
		feature_set[i] = np.array(sample_atom_list).reshape(-1, 1)
#
	feature_set = np.array(feature_set)
	input_shape = feature_set.shape
	#
	y = np.reshape(y, (len(y), len(y[0]), len(y[0])))
#
	early_stopping = EarlyStopping(patience=5)
	datasetsize = []
	loss_train = []
	loss_test = []
	rmsd_train = []
	rmsd_test = []
	for i in range(1): #100, 10000, 100
		i = 1000
		#model = cnn.generate_model_contact_map_1d(input_shape, (NUMBER_OF_AA2, NUMBER_OF_AA2))
		model = cnn.generate_model_aan_contact_map_1d(input_shape, (NUMBER_OF_AA2, NUMBER_OF_AA2))

		X = feature_set[:i]
		y_ = y[:i]

		X_train, X_test, y_train, y_test = train_test_split(X, y_, test_size=0.2)

		datasetsize.append(len(X))
		history = model.fit(X_train, y_train, batch_size=10, epochs=100, verbose=1, validation_data=(X_test, y_test), callbacks=[early_stopping])

		loss_train.append(history.history['loss'][len(history.history['loss']) - 1])
		loss_test.append(history.history['val_loss'][len(history.history['val_loss']) - 1])

		# All samples
		y_pred = model.predict(X)

		# Training and testing samples separately
		#y_pred_train = model.predict(X_train)
		#y_pred_test = model.predict(X_test)

		pca = PCA(n_components=3)

		# All samples
		coordinates_pred = [geo_distmat_to3d(elem) for elem in y_pred]
		coordinates_act = [geo_distmat_to3d(elem) for elem in y_]

		coordinates_pred_aligned = [geo_alignpoints(coordinates_act[i], coordinates_pred[i]) for i in range(len(y_))]

		rmsd = [sqrt(mean_squared_error(coordinates_pred_aligned[i], coordinates_act[i])) for i in range(len(y_)) ]

		# Training samples
		#coordinates_pred = [geo_distmat_to3d(elem) for elem in y_pred_train]
		#coordinates_act = [geo_distmat_to3d(elem) for elem in y_train]

		#coordinates_pred_aligned = [geo_alignpoints(coordinates_act[i], coordinates_pred[i]) for i in range(len(y_train))]

		#rmsd_train_ = [sqrt(mean_squared_error(coordinates_pred_aligned[i], coordinates_act[i])) for i in range(len(y_train)) ]

		#rmsd_train.append(np.mean(rmsd_train_))

		# Testing samples
		#coordinates_pred = [geo_distmat_to3d(elem) for elem in y_pred_test]
		#coordinates_act = [geo_distmat_to3d(elem) for elem in y_test]

		#coordinates_pred_aligned = [geo_alignpoints(coordinates_act[i], coordinates_pred[i]) for i in range(len(y_test))]

		#rmsd_test_ = [sqrt(mean_squared_error(coordinates_pred_aligned[i], coordinates_act[i])) for i in range(len(y_test)) ]
		
		#rmsd_test.append(np.mean(rmsd_test_))

	if False:
		plt.plot(datasetsize, loss_train)
		plt.plot(datasetsize, loss_test)
		plt.plot(datasetsize, rmsd_train)
		plt.plot(datasetsize, rmsd_test)
		plt.legend(['Train MSE Loss', 'Validation MSE Loss', 'Train RMSD', 'Validation RMSD'], loc='upper right')
		plt.title('1-D CNN Metrics vs data set size for MDS')
		plt.ylabel('MSE Loss (A^2), RMSD (A)')
		plt.xlabel('Data set size')
		plt.savefig('figures/ptndata_1dconv_summary_mds.png')
		plt.show()
		plt.clf()

	if False:
		plt.plot(datasetsize, loss_train)
		plt.plot(datasetsize, loss_test)
		plt.plot(datasetsize, rmsd_train)
		plt.plot(datasetsize, rmsd_test)
		plt.legend(['Train MSE Loss', 'Validation MSE Loss', 'Train RMSD', 'Validation RMSD'], loc='upper right')
		plt.title('1-D ANN Metrics vs data set size for MDS')
		plt.ylabel('MSE Loss (A^2), RMSD (A)')
		plt.xlabel('Data set size')
		plt.savefig('figures/ptndata_1dann_summary_mds.png')
		plt.show()
		plt.clf()

	if False:
		data = pd.DataFrame({'abs_loss': [history.history['loss']], 'abs_val_loss': [history.history['val_loss']]})
		data.to_csv('figures/ptndata_1dconv.csv')
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('1-D CNN Contact Map Metric')
		plt.ylabel('MSE Loss (A^2)')
		plt.xlabel('Epoch')
		plt.legend(['Training set', 'Validation set'], loc='upper left')
		plt.savefig('figures/ptndata_1dconv_abs_loss.png')
		plt.clf()

	if False:
		data = pd.DataFrame({'abs_loss': [history.history['loss']], 'abs_val_loss': [history.history['val_loss']]})
		data.to_csv('figures/ptndata_1dconv_ann.csv')
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('1-D CNN Contact Map Metric ANN')
		plt.ylabel('MSE Loss (A^2)')
		plt.xlabel('Epoch')
		plt.legend(['Training set', 'Validation set'], loc='upper left')
		plt.savefig('figures/ptndata_1dann_abs_loss.png')
		plt.clf()

	# For measuring the data size affects on the 1D metrics
	#return model, history, datasetsize, loss_train, loss_test, rmsd_train, rmsd_test

	min_rmsd_indx = np.argmin(rmsd)
	max_rmsd_indx = np.argmax(rmsd)

	min_rmsd_file = files[min_rmsd_indx]
	max_rmsd_file = files[max_rmsd_indx]

	min_rmsd = rmsd[min_rmsd_indx]
	max_rmsd = rmsd[max_rmsd_indx]


	# For measuring best/worst case scenario
	return model, history, files, rmsd, min_rmsd_indx, max_rmsd_indx, min_rmsd_file, max_rmsd_file, min_rmsd, max_rmsd, coordinates_pred_aligned, coordinates_act


def conv3d_tertiary_seq_rosetta_mse_dm(fdir='ptndata_10H/'):
#if True:
	start_time = time()
	total_samples = 1000
	validation_split = 0.2
#
	training_samples = int(total_samples * (1 - validation_split))
	validation_samples = int(total_samples * validation_split)
#
	# Path name for storing all of the data
	#fdir = 'ptndata_small/'
	#fdir = '/Users/ethanmoyer/Projects/data/ptn/ptndata_10H/'
	print('Loading files...')
	# Load all of the obj file types and sort them by file name
	files = getfileswithname(fdir, 'obj')
	files.sort()
#
	files = files[:total_samples]
#
	energy_scores = pd.read_csv(fdir + 'energy_local_dir.csv', index_col='file')
#
	files = [file for file in files if 'ptndata_10H/' + file in energy_scores.index]
#
	training_files = files[:training_samples]
	validation_files = files[training_samples:]
#
	# Index for the four main types of atoms and None that will indexed when looping through each entry
	atom_type = ['C', 'N', 'O', 'S', 'None']
	atom_type_data = pd.Series(atom_type)
	atom_type_encoder = np.array(pd.get_dummies(atom_type_data))
#
	print('Detemining positional atom types and smallest window size of the data ...')
	# Loop through each file and make a list of all of the atoms present.
#
	atom_pos, x_min, y_min, z_min, x_max, y_max, z_max = load_feature_dimensions(files, fdir)
#
	# Format the position specific atom list so it can be used as one-hot encoding in the network
	atom_pos_data = pd.Series(atom_pos)
	atom_pos_encoder = np.array(pd.get_dummies(atom_pos_data))
#
	# Initialize the feature set
	feature_set = np.array([[[[ [0] * (1 + len(atom_type) + len(atom_pos)) for i in range(x_min, x_max)] for j in range(y_min, y_max)] for k in range(z_min, z_max)] for q in range(1)])
#
	feature_set_ = np.array([[[[ [0] * (1 + len(atom_type) + len(atom_pos)) for i in range(x_min, x_max)] for j in range(y_min, y_max)] for k in range(z_min, z_max)] for q in range(validation_samples)])
#
	# Define input and output shape
	input_shape = feature_set.shape
	output_shape = (20, 20)
#
	#cnn = cnn()
	#model = cnn.generate_model_rosetta_mse(input_shape)
	model = cnn.generate_model_contact_map_3d(input_shape, output_shape)
#
	print('Generating validation data ...')
	# Load all of the objects into the feature set 
	feature_set, y_rosetta, y_mse, y_dm = sample_loader(validation_files, feature_set_, atom_type, atom_type_encoder, atom_pos, atom_pos_encoder, energy_scores, x_min, y_min, z_min, x_max, y_max, z_max, fdir)

	#early_stopping = EarlyStopping(patience=5, min_delta=0.1)

	print('Running model on training data...')
	history = model.fit(sample_gen(training_files, feature_set, atom_type, atom_type_encoder, atom_pos, atom_pos_encoder, energy_scores, x_min, y_min, z_min, x_max, y_max, z_max, fdir), steps_per_epoch=1,epochs =150, verbose=1, use_multiprocessing=True, validation_data=(feature_set, y_dm)) #, callbacks=[early_stopping]
	print('Time elapsed:', time() - start_time)

	data = pd.DataFrame({'abs_loss': [history.history['loss']], 'abs_val_loss': [history.history['val_loss']]})
	data.to_csv('figures/1crnAH10_dm.csv')
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('3-D CNN Conact Map Metric')
	plt.ylabel('MSE Loss (A^2)')
	plt.xlabel('Epoch')
	plt.legend(['Training set', 'Validation set'], loc='upper left')
	plt.savefig('figures/1crnAH10_dm_abs_loss.png')
	plt.clf()


# This function exports the protein as a .pdb file with 
def export(coords, file = None):
	atom_number = 1
	amino_acid_number = 1
	# If file is not assigned, generate temp file
	if file == None:
		file = tempfile.NamedTemporaryFile(dir = 'sample_ptn', mode = 'w+', suffix='.pdb').name
	# Open file and loop through all of the atoms in the protein and print all of their information to the file.
	# ethan: only write PDB files after structure is updated from the subsetting done, i.e. range of aa or chains
	with open(file, "w+") as f:	
		for atom_coords in coords:
			x = atom_coords[0]
			y = atom_coords[1]
			z = atom_coords[2]
			out = 'ATOM   %4d %-4s %3s %1s%4s    %8.3f%8.3f%8.3f  1.00  1.00           %s  \n' % (atom_number, 'CA', 'THR', 'A',amino_acid_number, x, y, z, 'CA')		
			f.write(out)		
			atom_number += 1
		amino_acid_number += 1
	return file

cnn = cnn()
#conv3d_tertiary_seq_rosetta_mse_dm('ptndata_10H/')
model, history, files, rmsd, min_rmsd_indx, max_rmsd_indx, min_rmsd_file, max_rmsd_file, min_rmsd, max_rmsd, coordinates_pred_aligned, coordinates_act = conv1d_primary_seq_dm()
