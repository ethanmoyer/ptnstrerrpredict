# Class used for storing individual protein structures between ptn.py and cnn.py
class data_entry:
	def __init__(de, mat, dm = None):
		de.mat = mat
		de.dm = dm

