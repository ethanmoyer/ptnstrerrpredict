	#ahmet: your ptn.ca() will have problems if we run into an amino acid that does not have CA. It is better to implement ptn.aa() above and make use of it here.

	#ahmet: fill this function.
def aa():
	aa_ = []
	lastchain = None
	lastresseq = None
	for atom in p.protein().get_atoms():
		atom_chain = Selection.unfold_entities(atom, 'C')[0].get_id()
		atom_res = Selection.unfold_entities(atom, 'R')[0].get_resname()
		if atom_chain != lastchain or atom_res != lastresseq:
			aa_.append({'chain': atom_chain, 'resseq': atom_res, 'atoms':[atom]})
			lastchain = atom_chain
			lastresseq = atom_res
		else:
			aa_[-1]['atoms'].append(atom)
	for a in aa_:

		for atom in a['atoms']:
			#to make sure we always have ca,n,c,cb atoms defined for each amino acid:
			atom_id = atom.get_id()
			if atom_id in ['CA', 'N', 'C', 'CB']:
				a[atom_id] = atom.get_coord()
				if atom_id != 'CA':
					a['CA'] = atom.get_coord()

	return aa_

def ca(p):
	return [a['CA'] for a in p.aa()]

		#TODO convert this into a numpy matrix before returning.

