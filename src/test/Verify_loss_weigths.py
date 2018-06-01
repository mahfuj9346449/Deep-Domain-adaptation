




import numpy as np 
import os
import keras
import tensorflow as tf 
import keras.backend as K 
import keras.backend.tensorflow_backend as KTF

from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
KTF.set_session(sess)

def BasicModel():
	i = Input(shape=(17,))
	out = Dense(100, activation="relu")(i)
	out1 = Dense(1, activation="sigmoid")(out)
	out2 = Dense(1, activation="sigmoid")(out)


	model = Model(inputs=[i], outputs=[out1, out2])
	return model




if __name__=="__main__":
	print("Start")
	np.random.seed(17)
	X = np.random.random((100, 17))
	Y = np.random.random((100,1))
	Y2 = (Y>0.5).astype(float)



	alpha = K.variable(0.1)
	beta = K.variable(1.0)

	my_optimizer = Adam(lr=1e-3)
	model = BasicModel()
	model.compile(loss=["mse", "binary_crossentropy"], 
		optimizer=my_optimizer, 
		loss_weights=[alpha, beta],
		metrics=["acc"])

	epochs = 10
	batch_size = 10
	for e in range(epochs):
		print("="*50)
		print("Epoch {}".format(e))
		print("="*50)
		for i in range(int(len(X)/batch_size)):
			x = X[i*batch_size:(i+1)*batch_size]
			y = Y[i*batch_size:(i+1)*batch_size]
			y2 = Y2[i*batch_size:(i+1)*batch_size]
			score = model.train_on_batch(x, [y, y2])
			print(score)
		
		K.set_value(alpha, K.get_value(alpha)*3)
		# if e ==2:
		# 	K.set_value(alpha, K.get_value(alpha)*2)
		# 	print(K.eval(alpha))
			# model.compile(loss=["mse", "binary_crossentropy"], 
			# 			optimizer=my_optimizer, 
			# 			loss_weights=[100., beta],
			# 			metrics=["acc"])

	# score = model.fit(X,[Y, Y2], epochs=100, batch_size=10)
	# print(score)
	# import ipdb;ipdb.set_trace()