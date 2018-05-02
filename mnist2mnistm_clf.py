# -*- coding: utf-8 -*-

# author: Lu LIN 15/01/2018
import os
import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization


from data_processing import DataLoader
# import ipdb; ipdb.set_trace()

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.5):
	'''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

	num_threads = os.environ.get('OMP_NUM_THREADS')
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

	if num_threads:
		return tf.Session(config=tf.ConfigProto(
			gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
	else:
		return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


KTF.set_session(get_session(gpu_fraction=0.45)) # NEW 28-9-2017
# ============================ Settings ============================
BATCH_SIZE = 128
NUM_CLASSES = 10
# EPOCHS = 10

# input image shape
# img_rows, img_cols = 28, 28

# Load data
# (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

def NN_model(input_shape, num_classes, useCNN=False):
	if useCNN:
		# A simple example of CNN 
		model = Sequential()
		model.add(Conv2D(32, kernel_size=(3, 3),
		                 activation='relu',
		                 input_shape=input_shape))
		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(num_classes, activation='softmax'))
		model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
		model.summary()
	else:
		model = Sequential()
		model.add(Dense(1000, input_dim= input_shape)) #1500
		# model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Dense(500)) #500
		# model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Dense(200)) #100
		# model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Dense(50))
		model.add(Activation('relu'))

		model.add(Dense(num_classes, activation='softmax'))
		model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
		# model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam',metrics=['accuracy'])

		model.summary()
	return model



# def preprocessing(x_train, y_train, x_test, y_test, useCNN=False):
# 	if useCNN:
# 		# Reshape data for CNN

# 		# Attention! we suppose that using Tensorflow (with image dim ordering 'channels_last' by default)
# 		x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
# 		x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
# 		input_shape = (img_rows, img_cols, 1)
# 		x_train = x_train.astype('float32')
# 		x_test = x_test.astype('float32')
		
# 		# Normalization (may be not necessary)
# 		x_train /= 255
# 		x_test /= 255

# 		print('x_train shape:', x_train.shape)
# 		print(x_train.shape[0], 'train samples')
# 		print(x_test.shape[0], 'test samples')

# 		# Convert class vectors to one-hot matrices
# 		y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
# 		y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
# 	else:
# 		x_train = x_train.reshape(x_train.shape[0], -1)
# 		x_test = x_test.reshape(x_test.shape[0], -1)
# 		input_shape = img_rows*img_cols
# 		x_train = x_train.astype('float32')
# 		x_test = x_test.astype('float32')
		
# 		# Normalization (may be not necessary)
# 		x_train /= 255
# 		x_test /= 255

# 		# x_train /= 25
# 		# x_test /= 25

# 		print('x_train shape:', x_train.shape)
# 		print(x_train.shape[0], 'train samples')
# 		print(x_test.shape[0], 'test samples')

# 		# convert class vectors to one-hot matrices
# 		y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
# 		y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
# 	return x_train, y_train, x_test, y_test,input_shape

def main(epochs=15, save_weights_path = "./Weights/mnist_weights.hdf5", mode = "train", num_classes = NUM_CLASSES, useCNN=False):
	# x_train, y_train, x_test, y_test,input_shape = preprocessing(X_train, Y_train, X_test, Y_test, useCNN=useCNN)
	dirname = "/".join(save_weights_path.split("/")[:-1])
	if not os.path.exists(dirname):
		os.makedirs(dirname)

	img_rows = 32
	img_cols = 32
	data_loader = DataLoader(img_res=(img_rows, img_cols))
	input_shape = (32,32, 3)


	if mode == "train":
		model = NN_model(input_shape, num_classes, useCNN=useCNN)
		checkpointer = ModelCheckpoint(filepath=save_weights_path, verbose=1, save_best_only=True, save_weights_only= True, monitor = 'val_acc') 
		model.fit(data_loader.mnist_X, keras.utils.to_categorical(data_loader.mnist_y, 10), epochs=epochs, shuffle=True, validation_split=0.05, batch_size= BATCH_SIZE, callbacks=[checkpointer])
		# model.save_weights(save_weights_path) 
		print("All done.")
	elif mode == "test":
		model = NN_model(input_shape, num_classes, useCNN=useCNN)
		model.load_weights(save_weights_path, by_name =True)
		score = model.evaluate(data_loader.mnistm_X, keras.utils.to_categorical(data_loader.mnistm_y, 10))
		print("Accuracy on test set: {}".format(score[1]*100))
		print("All done.")
	else:
		raise ValuerError("'mode' should be 'train' or 'test'.")



if __name__ =="__main__":
	import argparse
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--train', action='store_true')
	parser.add_argument('--test', action='store_true')
	parser.add_argument('--epochs', type=int, default=15)
	parser.add_argument('--weights_path', type=str, default="./test/mnist_weights_CNN.hdf5")
	parser.add_argument('--useCNN', action='store_true')

	args = parser.parse_args()
	if args.train:
		main(epochs=args.epochs, mode="train", save_weights_path = args.weights_path, useCNN=args.useCNN)
	elif args.test:
		main(epochs=args.epochs, mode="test", save_weights_path = args.weights_path, useCNN=args.useCNN)
	else:
		print("Mode should be 'train' or 'test'.")




