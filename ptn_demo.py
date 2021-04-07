# Load virtual environment
# source /Users/ethanmoyer/Projects/Packages/Python/venv/bin/activate

from random import random
from pprint import pprint
from copy import deepcopy
from os import getcwd
from importlib import reload

from pyrosetta import *
from pyrosetta.toolbox import *

from ptn import ptn

fdir = getcwd()

# Dicusss protein codes
# 1crn vs 1crnA vs 1crn0-10


# Define start and end positions of structure
start = int(random() * 3) + 7
end = start + 9


# Define object
p = ptn(f'1crnA{start}-{end}')


# Access identifier information
print(p.info)

print(p.id)
print(p.chain)
print(p.fromres)
print(p.tores)


# Locate the protein structure on your local machine
print(p.loc)


# Print information about the amino acids
pprint(p.aa())


# Show the first amino acid
pprint(p.aa()[0])


# Show the coordinates of the amino acids
print(p.aa_coords())


# Print the atom list
print(p.atom_list)


# We can filter different atom types--these are the main ones of interest.
print(p.c())
print(p.ca())
print(p.cb())
print(p.n())


# What metrics do we have on the protein?
# Distance matrix
print(p.generate_distance_matrix())
print(p.generate_distance_matrix_ca())
print(p.generate_distance_matrix_ca_cb())

# Pyrosetta energy, if you cannot get pyrosetta just comment these lines out
print(p.pyrossetta_energy_calculation())

scorefxn = get_fa_scorefxn()
print(scorefxn)


# Now for the object encoding
mat = p.ptn2grid(p.aa())
print(mat[0][0][0:10])
print(len(mat))
print(len(mat[0]))
print(len(mat[0][0]))
atom_obj = mat[0][0][0]


# Show the methods related to each grid object
mat_logical = p.ptn2grid(p.aa(), logical=True)
print(mat_logical)


# Let's just visualize a larger object
p_ = ptn(f'1crnA')
mat_logical = p_.ptn2grid(p.aa(), logical=True)
p_.visualize_grid(mat_logical)


# Create a copy of the object
p_decoy = p.messup()


# Now we can calculate distance from two protein structures
print(p.mse_contact_calc(p_decoy))


# There are methods to automate the generation of test data... look on ptn.py
