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

# For HPC 
# qsubi -pe smp 4 -l m_mem_free=5G -l h_vmem=5G
# screen -S ptn
# module load python/3.6
# source venv/bin/activate
# pip3 install --user --upgrade tensorflow
# python3.6 -i cnn.py

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

	y = [i for i in range(CUBIC_LENGTH_CONSTRAINT) for j in range(x_min, x_max) if (np.array(mat[j][i]) != 0.0).any()]
	y_min = min(y)
	y_max = max(y)

	z = [i for i in range(CUBIC_LENGTH_CONSTRAINT) for j in range(x_min, x_max) for k in range(y_min, y_max) if (np.array(mat[j][k][i]) != 0.0).any()]
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
		model.add(Conv3D(filters=6, kernel_size=16, strides=(1, 1, 1), padding="same", input_shape=input_shape[1:]))
		model.add(BatchNormalization())
		model.add(Dense(16, activation='relu'))
		model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2))
		model.add(Conv3D(filters=6, kernel_size=16, strides=(1, 1, 1), padding="same", input_shape=input_shape[1:]))
		model.add(BatchNormalization())
		model.add(Dense(32, activation='relu'))
		model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2))
		model.add(Conv3D(filters=6, kernel_size=16, strides=(1, 1, 1), padding="same", input_shape=input_shape[1:]))
		model.add(BatchNormalization())
		model.add(Dense(64, activation='relu'))
		model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2))
		model.add(Dropout(0.2))
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


def sample_gen(files, fdir='ptndata_10H/'):
	for q, file in enumerate(files):
		entry = pickle.load(open(fdir + file, 'rb'))
		a = grid2logical(entry.mat)
		b = grid2atomtype(entry.mat)
		c = grid2atom(entry.mat)
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
		yield (feature_set, np.array(y))


def sample_loader(files, samples, fdir='ptndata_10H/'):
	feature_set_ = np.array([[[[ [0] * (1 + len(atom_type) + len(atom_pos)) for i in range(x_min, x_max)] for j in range(y_min, y_max)] for k in range(z_min, z_max)] for q in range(samples)])
	y = []
	for q, file in enumerate(files):
		entry = pickle.load(open(fdir + file, 'rb'))
		a = grid2logical(entry.mat)
		b = grid2atomtype(entry.mat)
		c = grid2atom(entry.mat)
		dm_output = entry.dm
		# rosetta_score, mse_score
		#y = dm_output
		#y = np.reshape(y, (len(y), len(y[0][0]), len(y[0][0])))
		#y = y.astype(float)
		y.append(energy_scores.loc['ptndata_10H/' + file]['rosetta_score'])
		for i in range(len(feature_set[0])):
			for j in range(len(feature_set[0][0])):
				for k in range(len(feature_set[0][0][0])):
					
					feature_set_[q][i][j][k] = [a[x_min + i][y_min + j][z_min + k]] + b[x_min + i][y_min + j][z_min + k].tolist() + c[x_min + i][y_min + j][z_min + k].tolist()
	y = np.array(y)
	y = y.reshape(-1,1)		
	return (feature_set_, y)


start_time = time()
total_samples = 2
validation_split = 0.2

training_samples = int(total_samples * (1 - validation_split))
validation_samples = int(total_samples * validation_split)

# Path name for storing all of the data
fdir = 'ptndata_10H/'
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

# Define input and output shape
input_shape = feature_set.shape
#output_shape = y.shape

cnn = cnn()
model = cnn.generate_model(input_shape)
#model = cnn.generate_model_contact_map(input_shape, output_shape)

print('Running model ...')
# Load all of the objects into the feature set 

history = model.fit(sample_gen(training_files, fdir), epochs = 10, verbose=1,use_multiprocessing=True) #, validation_data=sample_loader(validation_files, validation_samples, fdir)
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