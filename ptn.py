# source /Users/ethanmoyer/Projects/Packages/Python/venv/bin/activate
# python3.7 -i ptn.py
# Working off of /Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/Bio/PDB/

from Bio.PDB import *

from pyrosetta import *
from pyrosetta.toolbox import *

from scipy.spatial import distance_matrix
from sklearn.metrics import mean_squared_error 

import matplotlib.pyplot as plt
import numpy as np
import os
import re
import random
import tempfile
import pandas as pd

import pickle

import copy

from data_entry import data_entry
from grid_point import grid_point
from equ import calculate_atom_occupacy, find_nearest_atom, find_atom_distances

from ptn_io import isfileandnotempty, getfileswithname
from geo import geo_move2position, geo_rotatebyangles_linear_alg, get_centriod, geo_generate_dihedral_angles

# Tryptophan (largest amino acid) = 0.67 nm in diameter 6.7 angstroms -> 7 A
# For 10 Tryptophan, 70 Angstroms x 70 Angstroms x 70 Angstroms
# Poole C and F J Owens, 'Introduction to Nanotechnology' Wiley 2003 p 315.
CUBIC_LENGTH_CONSTRAINT = 70

from enum import Enum
class Score(Enum):
	rosetta = 0
	mse_dm = 1
	dm = 2

class ptn:
	def __init__(p, info, chain = None, fromres = None, tores = None):	
		# Set protein_ to 0 by default	
		p.protein_ = 0
		
		# Chain ID
		p.chain = chain

		# If a file is given in place of id, load the file and set the id equal to the id in the file name.
		if isfileandnotempty(info):			
			p.loc = info
		else:
			# 1crnA 5-10'
			info = re.match('(ptndata/)?(?P<id>[\dA-Za-z0-9]{4})(?P<chain>[A-Z0-9])?\s*((?P<fromres>\d+)-(?P<tores>\d+))?', info)

			p.info = info.group()
			p.id = info.group('id')
			p.chain = info.group('chain')

			# Used for subsetting the residues
			p.fromres = info.group('fromres')
			if p.fromres is not None:
				p.fromres = int(p.fromres)

			p.tores = info.group('tores')
			if p.tores is not None:
				p.tores = int(p.tores)

			# If incorrect protein identifier is passed
			if p.id is None and p.chain is None and p.fromres is None and p.tores is None:
				print("Please pass a valid protein identifier into the constructor as (id){chain}{from-to}")
				quit()

			# Download structure if a file doesn't exist
			print(p.id)

			pdir = '/Users/data/pdb/' + p.id[1:3]

			pdbl = PDBList()
			p.loc = pdbl.retrieve_pdb_file(p.id, pdir = pdir)

		# Set amino acids to none for easier reference.
		p.aa_ = None


	# Represents the entire structure loaded from pdb
	def protein(p): 
		if p.protein_ == 0:

			if '.pdb' in p.loc:
				parser = PDBParser()
			elif '.cif' in p.loc:
				parser = MMCIFParser()

			p.protein_ = parser.get_structure('id', p.loc)

			# If chain is not given, use all of them. Otherwise, use given.
			if p.chain == None:
				p.protein_ = p.protein_[0]
			else:
				p.protein_ = p.protein_[0][p.chain]

		return p.protein_;


	# Work on this function... Given a protein and a helix identifier, find all ranges in the protein which are identified by that helix
	def find_helix(p, helix='H'):
		return 0


	# This function returns all of the amino acids of a protein with data about each of the atoms.
	def aa(p):
		# If amino acid list has been previously assigned, return it.
		if p.aa_ is not None:
			return p.aa_

		# Otherwise, start building new amino acid list.
		aa_ = []
		p.atom_list = []
		# Keep track of last chain and resseq
		lastchain = None
		lastresseq = None

		# Residue number in the protein
		res_number = 0

		# Loop through all of the atoms in the protein structure
		for atom in p.protein().get_atoms():
			atom_chain = Selection.unfold_entities(atom, 'C')[0].get_id()
			atom_res = Selection.unfold_entities(atom, 'R')[0]
			
			#ahmet: if p.fromres is used && atom_res < p.fromres || atom_res > p.tores; continue; end
			if (p.fromres is not None and p.tores is not None) and (res_number < p.fromres or res_number > p.tores) and (atom_chain != lastchain or atom_res != lastresseq):
				res_number += 1
				continue
			
			# If the atom is from a unique residue or chain, append the atom as a new entry to the amino acid list
			if atom_chain != lastchain or atom_res != lastresseq: 
				# 
				aa_.append({'chain': atom_chain, 'resseq': atom_res, 'atoms':[atom]})
				lastchain = atom_chain
				lastresseq = atom_res
				p.atom_list.append(atom.get_name())
				res_number += 1

			# If it is not from a unqiue residue, append it to the previous residue entry.
			else:
				aa_[-1]['atoms'].append(atom)
				p.atom_list.append(atom.get_name())

		# Loop through all of the residues in the protein.
		for a in aa_:

			# Loop through all of the atoms in each of the residues
			for atom in a['atoms']: 
				atom_id = atom.get_id()

				# If the atom ID is either 'CA', 'CB', 'C', or 'N,' store their coordinates
				if atom_id in ['CA', 'N', 'C', 'CB']:
					a[atom_id] = atom.get_coord()

			# Check if there is at least one alpha carbon for the current amino acid.
			if 'CA' not in a:
				print('There was not an alpha carbon for this amino acid', a['resseq'].get_resname(), '\nThe first atom will be used instead.')
				a['CA'] = a['atoms'][0].get_coord()

			# Check if there is at least one beta carbon for the current amino acid.
			if 'CB' not in a:
				print('There was not a beta carbon for this amino acid', a['resseq'].get_resname(), "\nThe alpha carbon will be used instead.")
				a['CB'] = a['CA']

			# Check if there is at least one nitrogen for the current amino acid.
			if 'N' not in a:
				print('There was not a nitrogen for this amino acid', a['resseq'].get_resname(), "\nThe alpha carbon will be used instead.")
				a['N'] = a['CA']

			# Check if there is at least one  carbon for the current amino acid.
			if 'C' not in a:
				print('There was not a carbon for this amino acid', a['resseq'].get_resname(), "\nThe alpha carbon will be used instead.")
				a['C'] = a['CA']

		# Set object's amino acid list for easier reference.
		p.aa_ = aa_

		return aa_


	# This function returns all of the atomic coordinates for each amino acid.
	def aa_coords(p):
		return np.array([atom.get_coord() for aa in p.aa() for atom in aa['atoms']])


	# This function only returns the alpha carbon coordinates.
	def ca(p):
		return np.array([a['CA'] for a in p.aa()])


	# This function only returns the beta carbon coordinates.
	def cb(p):
		return np.array([a['CB'] for a in p.aa()])


	# This function only returns the nitrogen atom coordinates.
	def n(p):
		return np.array([a['N'] for a in p.aa()])


	# This function only returns the carbon atom coordinates.
	def c(p):
		return np.array([a['C'] for a in p.aa()])


	# Export pdb and run pyrosseta to calculate energy
	def pyrossetta_energy_calculation(p, file = None):
		return p.energy_calc(p.export(file))


	# Calculate energy given pdb file using pyrosseta. Return score and location of pdb file.
	def energy_calc(p, pdb):
		# Initialize pyrosseta
		init()

		# Cleaning pdb structure creates .clean.pdb in same directory from which the pdb was found
		cleanATOM(pdb)

		# Generate pose from cleaned pdb file
		pose = pose_from_pdb(re.sub('.pdb','.clean.pdb', pdb))

		# Generate scoring function
		scorefxn = get_fa_scorefxn()

		# Apply scoring function on structure
		score = scorefxn(pose)
		return score, pdb


	def generate_distance_matrix(p):
		return(distance_matrix(p.aa_coords(), p.aa_coords()))


	def mse_contact_calc(p, p_):
		return(mean_squared_error(p.generate_distance_matrix(), p_.generate_distance_matrix()))


	# This function exports the protein as a .pdb file with 
	def export(p, file = None):
		atom_number = 1
		amino_acid_number = 1

		# If file is not assigned, generate temp file
		if file == None:
			file = tempfile.NamedTemporaryFile(dir = 'tempfiles', mode = 'w+', suffix='.pdb').name
		
		# Open file and loop through all of the atoms in the protein and print all of their information to the file.
		# ethan: only write PDB files after structure is updated from the subsetting done, i.e. range of aa or chains
		with open(file, "w+") as f:
			for aa in p.aa():
			
				for atom in aa['atoms']:
					atom_coords = atom.get_coord()
					x = atom_coords[0]
					y = atom_coords[1]
					z = atom_coords[2]
					atom_name = atom.get_name()
					# pdb file format fprintf(fid, 'ATOM   %4d %-4s %s %s%5s   %8.3f%8.3f%8.3f  1.00  1.00           %s  \n', ... atomi, upper(a.resname), chain, ptn_num2icoded(resseqs(i)), X(i,1), X(i,2), X(i,3);
					out = 'ATOM   %4d %-4s %3s %1s%4s    %8.3f%8.3f%8.3f  1.00  1.00           %s  \n' % (atom_number, atom_name, aa['resseq'].get_resname(), aa['chain'], amino_acid_number, x, y, z, atom_name[:1])

#'ATOM  %4d %-4s %3s %1s%4s    %8.3f%8.3f%8.3f  1.00  1.00           %s  \n'
			
					f.write(out)
					
					atom_number += 1
				amino_acid_number += 1

		return file


	# Generate fasta of the data
	def fasta(p):
		with open("seq.fasta", "w") as f:
			print(''.join([res.get_resname() for res in p.protein().get_residues()][:10]), file = f)


	# Randomly orientate the positions x,y,z positions of atoms in a 3D protein structure 
	def messup(p): 

		# Create new object, copy of p and return that altered copy
		p_ = copy.deepcopy(p)

		# Multiplier for scaling delta
		multiplier = 1

		# Loop through all atoms in all amino acids and move each of them by a randomly generated delta value
		for aa in p_.aa():
			for atom in aa['atoms']:
				for i in range(3):
					# Generate small delta between -1 and 1 and add that to all of the coordinates
					delta = (random.random() - 0.5) * multiplier
					atom.get_coord()[i] = atom.get_coord()[i] + delta

		return p_


	# Creates logical 3D structure based on the positions of atoms in a protein's 3D structure
	def ptn2grid(p, amino_acids, center = [CUBIC_LENGTH_CONSTRAINT/2, CUBIC_LENGTH_CONSTRAINT/2, CUBIC_LENGTH_CONSTRAINT/2], angles = None): 

		# Make a copy of p and alter that.
		p_ = copy.deepcopy(p)

		# If angles are provided, rotate the figure. Otherwise, do nothing.
		if angles is not None:
			atoms = geo_rotatebyangles_linear_alg(p_.aa_coords(), angles)
		else:
			atoms = p_.aa_coords()

		# Regardless, shift structure to the center of the 3D window
		atoms_shifted = geo_move2position(atoms, center)

		atom_number = 0

		# Reassign shifted coordinates back to p_ structure
		for aa in p_.aa():
			for atom in aa['atoms']:
				for j in range(3):
					atom.get_coord()[j] = atoms_shifted[atom_number][j]

				atom_number += 1

		# ethan: there's a math domain error here every so often
		# Generate dihedral angles for the structure
		#dihedreal_angles = geo_generate_dihedral_angles(p_.aa_coords())

		# Length constant representing the size of the 3D window
		a = CUBIC_LENGTH_CONSTRAINT

		# Add resolution
		resolution = 1

		# Empty 3D window 
		logical_mat = np.zeros((a, a, a))

		# Initialize a x a x a grid of points with objects for storing data
		mat = [[[grid_point(coords=[i + 0.5, j + 0.5, k + 0.5]) for i in range(a)] for j in range(a)] for k in range(a)]

		atom_number = 0

		# Loop through all atoms in all amino acids and place a logical 1 at all of the coordinate tuples
		for aa in p_.aa():
			for atom in aa['atoms']:

				# Coordinates of each atom divided by the resolution
				x = int(round(atom.get_coord()[0] / resolution))
				y = int(round(atom.get_coord()[1] / resolution))
				z = int(round(atom.get_coord()[2] / resolution))

				# Print error if any of the points are negative--they never should be.
				if x < 0 or y < 0 or z < 0:
					print(f'Error at atom number %d. One of its coordinates is negative: (%d, %d, %d)', atom_number, x, y, z)
					quit()

				# Assign a logical 1 at the position of the atom
				logical_mat[x, y, z] = 1

				# Store atom and residue names
				mat[x][y][z].atom = atom.get_name()
				mat[x][y][z].coords = atom.get_coord()
				mat[x][y][z].aa = aa['resseq'].get_resname()

				# Store dihedral angles at all atoms except for the first one and last two
				#if (atom_number != 0 and atom_number < len(p_.aa()) - 2):
				#	mat[x][y][z].diangle = dihedreal_angles[atom_number - 1]

				# Create distance vector from the atoms points to all of the other points. Normalize those into vectors and find and store the minimum distance.
				

				

				# Count the number of atoms within a certain threshold distance.

				atom_distances = find_atom_distances(atom.get_coord(), atoms_shifted)

				mat[x][y][z].atoms_within_threshold = len(atom_distances[atom_distances <= mat[x][y][z].threshold])

				atom_number += 1

		# Calculate occupancy for each grid point
		for x in range(a):
			for y in range(a):
				for z in range(a):

					coords = mat[x][y][z].coords

					distance_to_nearest_atom, nearest_atom_number = find_nearest_atom(coords, atoms_shifted)

					mat[x][y][z].distance_to_nearest_atom = distance_to_nearest_atom

					nearest_atom = p.atom_list[nearest_atom_number][:1]

					mat[x][y][z].nearest_atom = nearest_atom


					mat[x][y][z].occupancy = calculate_atom_occupacy(nearest_atom, distance_to_nearest_atom)

					#print(mat[x][y][z].coords)
					#print(nearest_atom)
					#print(distance_to_nearest_atom)
					#print(mat[x][y][z].occupancy)
					#if z == 5:
					#	quit()

		return mat


	# Generate random data by messing up the original protein structure, calculating the energy using pyrosseta, and rotating it a random amount. Then save the score and matrix the data directory.
	def generate_decoy_messup_scores(p, n = 10, native_rate = 0, start = 0, score_types = [Score.rosetta, Score.mse_dm, Score.dm], fdir = 'ptndata/', decoys = None):
		if decoys is not None:
			n = len(decoys)

		for i in range(start, n + start):
			if decoys is None:
				p_ = copy.deepcopy(p)
				if random.random() > native_rate:
					p_ = p_.messup()
				file = fdir + p_.info + '_mess' + str(i) + '.pdb'
				p_.export(file)
			else:
				p_ = decoys[i]
				file = p_.loc

			scores = pd.DataFrame()

			if Score.rosetta in score_types:
				score0, _ = p_.energy_calc(file)
				scores['rosetta_score'] = [score0]
			if Score.mse_dm in score_types:
				score1 = p_.mse_contact_calc(p)
				scores['mse_dm_score'] = [score1]
			if Score.dm in score_types:
				score2 = p_.generate_distance_matrix()
				scores['dm_score'] = [score2]

			mat = p_.ptn2grid(p_.aa(), angles = [random.random() * 360, random.random() * 360, random.random() * 360])
			p_.save_data(mat, scores = scores, file = file, fdir = fdir,energy_file = fdir + 'energy.csv')


	# This function stores a data_entry consisting of the 3D matrix with its relative score
	def save_data(p, mat, scores = None, file = None, fdir = 'ptndata/', energy_file = 'ptndata/energy.csv'):
		# If no file is provided, create a temporary named file in the ptndata directory. Otherwise if pdb is in the file name, create file named the same as the .pdb file as an obj file.
		if file == None:
			file = tempfile.NamedTemporaryFile(dir = 'ptndata', mode = 'w+', suffix='.obj').name
		elif 'clean.pdb' in file:
			file = re.sub('.clean.pdb','.obj', file)
			file = re.sub('tempfiles/', fdir, file)
		elif 'pdb' in file:
			file = re.sub('.pdb','.obj', file)
			file = re.sub('tempfiles/', fdir, file)
		else:
			print('Please provide a correct file type')
			quit()

		# Create score entry using file name without the extension and the score of the protein. Append this to the csv file storing scores.
		dm = None
		score_entry = pd.DataFrame({'file': [file]})
		for i in range(len(scores.columns)):
			if scores.columns[i] == 'dm_score':
				dm = scores['dm_score']
				continue
			score_entry[scores.columns[i]] = scores[scores.columns[i]]

		score_entry.to_csv(energy_file, mode = 'a', header = False, index = False)

		# Create a data entry of the given matrix and dump it as aa .obj file.
		data_entry_ = data_entry(mat, dm = dm) 
		filehandler = open(file, 'wb') 
		pickle.dump(data_entry_, filehandler)


	# This function visualizes a grid produced by ptn2grid function.
	def visualize_grid(p, mat):
		# Initializes plot
		fig = plt.figure()

		# Create 3D plot with a x a x a dimensions
		ax = plt.axes(projection='3d')
		ax.set_xlim(0, CUBIC_LENGTH_CONSTRAINT); 
		ax.set_ylim(0, CUBIC_LENGTH_CONSTRAINT); 
		ax.set_zlim(0, CUBIC_LENGTH_CONSTRAINT);

		# Set labels
		ax.set_xlabel('$X$', fontsize=20)
		ax.set_ylabel('$Y$', fontsize=20)
		ax.set_zlabel('$Z$', fontsize=20)

		# Add protein data
		xdata, ydata, zdata = np.where(mat == 1)
		ax.scatter3D(xdata, ydata, zdata);

		# Show plot
		fig.show()


	# This function compares a normal protein to a rotated protein using the visualize_grid function.
	def test_rotation(p):
		# Generate and visualize normal protein
		mat0 = p.ptn2grid(p.aa())
		p.visualize_grid(mat0)

		# Generate and visualize protein rotated 90 and 180 degrees about the x axis
		mat1 = p.ptn2grid(p.aa(), angles = [90,0,0])
		p.visualize_grid(mat1)
		mat2 = p.ptn2grid(p.aa(), angles = [180,0,0])
		p.visualize_grid(mat2)


	def load_decoys(p, fdir = 'ptndata/'):
		# Load all of the obj file types and sort them by file name
		files = getfileswithname(fdir, [p.id, 'clean.pdb'])
		p_list = []
		for file in files:
			file = 'ptndata/' + file
			p_list.append(ptn(file))

		return(p_list)


for i in range(1000):

	start = int(random.random() * 8) + 7
	end = start + 4

	p = ptn(f'1crnA{start}-{end}')

	p.generate_decoy_messup_scores(1, native_rate = 0.1, start = 100 + i, fdir = 'ptndata_small/')


# Below is script
if (False):

	ids = pd.read_csv('training.txt').values

	for id in ids:
		p = ptn(id[0] + 'A0-10')
		p.generate_decoy_messup_score_mse_mat(100, '/Users/data/ptnstrerrpredict/ptndata/')

	for p_decoy in p.load_decoys():
		mat = p_decoy.ptn2grid(p_decoy.aa())
		score0 = p_decoy.mse_contact_calc(p)
		score1, _ = p.energy_calc()
		p_decoy.save_data(mat, pd.DataFrame({'rosetta_score': [score0], 'mse_score': [score1]}), file = p_decoy.loc, energy_file = fdir + '/energy.csv')

	# Use data with one alpha helix
	# DNA structure
	p = ptn('103dB')


	#ahmet: test ptn() for an example NMR file.

	#ahmet: test ptn() for an example protein having multiple chains.



