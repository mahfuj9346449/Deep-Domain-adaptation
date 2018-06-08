import os
import dill
import numpy as np
		
class _DLalgo(object):
	"""
	Deep Learning algorithms classs

	"""
	def __init__(self, **kwargs):
		super(_DLalgo, self).__init__()
		##### Set up the other attributes
		for key in kwargs:
			setattr(self, key, kwargs[key])

	def __checktype(self, A):
		key_to_be_purge = []
		for key in A:
			if not type(A[key]) in [list, dict, int, float, str, tuple, np.ndarray, bool]:
				# A.pop(key)
				key_to_be_purge.append(key)
			else:
				pass
		print("Purging {} keys (in order to save config): {}.".format(len(key_to_be_purge), key_to_be_purge))
		for key in key_to_be_purge:
			A.pop(key)
		print("+ Done.")
		return A

	def save_config(self, save2path="./test.dill", verbose=False):
		"""
		Save config at the end (before training) !

		"""
		dirpath = "/".join(save2path.split("/")[:-1])

		if not os.path.exists(dirpath):
			os.makedirs(dirpath)
		

		# A shallow copy of self.__dict__
		normal_attrs = dict(self.__dict__)
		normal_attrs = self.__checktype(normal_attrs)
	
		print("Saving {} class attributes to file {}...".format(len(normal_attrs), save2path))
		with open(save2path, "wb") as file:
			dill.dump(normal_attrs, file)
		if verbose:
			print("Normal attributes are: {}".format(normal_attrs))
		print("+ Done.")
		# print(len(self.__dict__))

	def load_config(self, from_file="./test.dill", verbose=False):
		"""
		It's important to load config BEFORE build_model !


		TODO: add warning message about overwritting attributes
		"""
		print("Loading class attributes from file {}...".format(from_file))
		with open(from_file, "rb") as file:
			kwargs = dill.load(file)
		## init default attributes
		if verbose:
			print("Number of attributes: {}".format(len(kwargs)))
		self.__init__(**kwargs) # TODO !!!

		## init attributes that are created in class functions
		for key in kwargs:
			setattr(self, key, kwargs[key])

		print("+ Done.")

	def print_config(self, return_message=True):
		message = ""
		message = message+"="*50+"\n"
		message = message+" "*20+"Config"+"\n"
		message = message+"="*50+"\n"
		for key in self.__dict__:
			message = message+"{}: {}".format(key, self.__dict__[key])+"\n"
			
		message = message+"="*50+"\n"
		print(message)
		if return_message:
			return message

class TestAlgo0(_DLalgo):
	"""docstring for TestAlgo0"""
	def __init__(self, arg, **kwargs):
		super(TestAlgo0, self).__init__()
		self.arg = arg
		for key in kwargs:
			setattr(self, key, kwargs[key])
	def print(self):
		self.print_config()

if __name__ == "__main__":
	print("Start")


	dummy_algo = TestAlgo0(20, a=17)
	dummy_algo.load_config()
	dummy_algo.print()
	# dummy_algo.save_config()