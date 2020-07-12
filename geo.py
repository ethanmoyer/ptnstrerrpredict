# Geometrical functions that can be applied to points in R3

# Euler angles to roation matrix
from scipy.spatial.transform import Rotation as R

# Find dihedral angles
from pyrosetta.toolbox.numpy_utils import calc_dihedral

import numpy as np

# Centriod of coordinates
def get_centriod(coords):
	return np.mean(coords, axis = 0) / len(coords)
	retu


# Moves coordinates to arbitary point in space. Default point is center of coordinates.
def geo_move2position(coords, pos = None):

	if pos == None:
		pos = get_centriod(coords)

	x = pos[0]
	y = pos[1]
	z = pos[2]

	# Translation matrix
	tran_mat_pos = np.matrix([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

	# Adds 4th column of ones
	coords = np.column_stack([coords, np.ones((len(coords),1))])

	# Returns translated coordinates
	return np.array(coords * tran_mat_pos.T)[:, :3]


# Given a set of atom coordinates (3-column numpy matrix), rotate the atoms around their center of mass. Version 1.
def geo_rotatebyangles_linear_alg(coords, angles):
	# Calculate centriod matrix
	centroid = get_centriod(coords)

	# Define centriod positions
	x = centroid[0]
	y = centroid[1]
	z = centroid[2]

	# Create translation matrix to origin
	tran_mat_origin = np.matrix([[1, 0, 0, -x], [0, 1, 0, -y], [0, 0, 1, -z], [0, 0, 0, 1]])

	# Create translation matrix away from origin
	tran_mat_centriod = np.matrix([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

	# Create rotation matrix about origin
	r = R.from_euler('xyz', angles)
	rot_mat_origin = r.as_matrix()

	# Convert rotation matrix about origin to 4x4
	rot_mat_origin_4 = np.zeros((4,4))
	for i in range(3):
		for j in range(3):
			rot_mat_origin_4[i][j] = rot_mat_origin[i][j]

	rot_mat_origin_4[3][3] = 1

	# Convert coords into a nx4 matrix
	coords_4d = np.column_stack([coords, np.zeros((len(coords),1))])

	# Perform rotation. T(x, y) * R * T(-x, -y) * P
	coords_rotated = tran_mat_centriod @ rot_mat_origin_4 @ tran_mat_origin @ coords_4d.T

	# Return everything except for fourth column
	return np.array(coords_rotated[:3,]).T


# Given a set of atom coordinates (3-column numpy matrix), rotate the atoms around their center of mass. Version 2.
# This function doesn't seem to work... [90,0,0]/[-90,0,0] test fails.
def geo_rotatebyangles_simple_trasnlation(coords, angles):
	# Calculate centriod matrix
	centroid = np.mean(coords, axis = 0) / len(coords)

	# Define centriod positions
	x = centroid[0]
	y = centroid[1]
	z = centroid[2]

	# ethan: Change to simple translation as opposed to two translation 
	# numpy.substract()
	coords = coords - centroid

	# Create rotation matrix about origin
	r = R.from_euler('xyz', angles)
	rot_mat_origin = r.as_matrix()

	# Perform rotation
	coords_rotated = coords @ rot_mat_origin.T

	# Return everything except for fourth column
	return coords_rotated + centroid


# Calculate dihedral angles of coordinates in R3
def geo_generate_dihedral_angles(coords):
	return [calc_dihedral(coords[i:i + 4]) for i in range(len(coords) - 3)]


