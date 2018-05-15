



import os, sys
import numpy as np 
import dill

class MyConfig(object):
	"""docstring for MyConfig"""
	def __init__(self):
		super(MyConfig, self).__init__()

	def save_config(self, save2path="./my_config_test.pickle"):

		with open(save2path, "wb") as file:
			dill.dump(self.__dict__, file)
		return
	def print_config(self):
		print(self.__dict__)

class MyClass(MyConfig):
	"""docstring for MyClass"""
	def __init__(self, **kwargs):
		super(MyClass, self).__init__()
		self.a = kwargs["a"]
		# self.b = None#kwargs["b"]
		# self.c = None#kwargs["c"]
		print(kwargs)
	def create_attr(self, b, c):
		self.b = b
		self.c = c
		self.fun = lambda x:self.Myfun(x)
	def set_config(self, **kwargs):

		print(kwargs)
		for key in kwargs:
			setattr(self, key, kwargs[key])

	def Myfun(self, x):
		return 2*x

	def load_config(self, filepath):
		with open(filepath, "rb") as file:
			A = dill.load(file)
		print(A)
		self.__init__(**A)
		self.set_config(**A)

class TestA(object):
	"""docstring for TestA"""
	def __init__(self, arg, **kwargs):
		super(TestA, self).__init__()
		self.arg = arg
		
	def build_model(self):
		self.b = 30
		self.model1 = lambda x:self.myfunc(self.b*x)


	def myfunc(self, x):
		return 2*x

	def save_config(self, save2path="./testA.dill"):

		with open(save2path, "wb") as file:
			dill.dump(self.__dict__, file)

	def load_config(self, from_path="./testA.dill"):

		with open(from_path, "rb") as file:
			kwargs = dill.load(file)

		for key in kwargs:
			setattr(self, key, kwargs[key])
	def print_config(self):
		print(self.__dict__)

if __name__ =="__main__":
	print("Start")

	# obj = MyClass(a=1)
	# obj.create_attr(4,5)
	# obj.print_config()
	# obj.load_config("./my_config_test.pickle")
	# obj.print_config()
	# obj.save_config()
	# import ipdb; ipdb.set_trace()

	# with open("./my_config_test.pickle", "rb") as file:
	# 	A = dill.load(file)
	# print(A)
	# import ipdb; ipdb.set_trace()


	obj = TestA(100, test=17, test2=27)
	obj.build_model()
	obj.print_config()
	print("="*50)
	obj.load_config()
	obj.print_config()
	# obj.save_config()
	print(obj)