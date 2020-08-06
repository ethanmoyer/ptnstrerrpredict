from numpy import exp, power, delete, sqrt
import numpy as np

atom_waals_radii = {'C': 0.170, 'O': 0.152, 'N': 0.155, 'S': 0.180}

def calculate_atom_occupacy(nearest_atom, distance_to_nearest_atom):
	return 1 - exp(-power((atom_waals_radii[nearest_atom] / distance_to_nearest_atom), 12))


def find_nearest_atom(position, atoms):

	atom_distances = find_atom_distances(position, atoms)

	min_distance = np.min(atom_distances)
	min_distance_index = atom_distances.tolist().index(min_distance)

	return min_distance, min_distance_index


def find_atom_distances(position, atoms):
	distnace_vector = atoms - position

	atom_distances = sqrt(distnace_vector[:,0]**2 + distnace_vector[:,1]**2 + distnace_vector[:,2]**2)

	return atom_distances