

import numpy as np 
import os
import keras

from keras.layers import Dense, Input
from keras.models import Sequential, Model
import keras.backend as K

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.utils import plot_model

def get_session(gpu_fraction=0.5):
	'''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

	num_threads = os.environ.get('OMP_NUM_THREADS')
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

	if num_threads:
		return tf.Session(config=tf.ConfigProto(
			gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
	else:
		return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session(gpu_fraction=0.2))


def Simple_model():
	i = Input(shape=(100,))
	a = Dense(10, activation="relu")(i)
	a = Dense(17)(a)

	model = Model(inputs=[i], outputs=[a])
	return model


if __name__ == "__main__":
	print("Start")
	a = K.zeros(10)
	print(K.eval(a))

	############################################# 
	# Plot model using graphviz and pydot package
	#############################################

	# model = Simple_model()
	# model.summary()
	# plot_model(model, to_file="model.png")
	model = Simple_model()
	TC = keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=0, write_graph=True, write_images=False)

	TC.set_model(model)
	# for i in range(10):
	# 	acc = 0.0
	# 	loss = 1.0
	# 	val_loss = 1.0
	# 	val_acc = 0.0
	# 	# then after each epoch
	# 	logs = {'acc': acc, 'loss': loss, 'val_loss': val_loss, 'val_acc': val_acc}

	# 	TC.on_epoch_end(i, logs)
		# acc is needed when doing categorisations not regression,
		# val_loss & val_acc are optional, they would be provided by fit if you used a validation split or a validation dataset.

	# and when finished

	# TC.on_train_end('_')