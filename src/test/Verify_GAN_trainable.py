

import numpy as np
import os
import keras
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.layers import Dense, Input, Flatten, Reshape
from keras.models import Model

def get_session(gpu_fraction=0.5):
	'''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

	num_threads = os.environ.get('OMP_NUM_THREADS')
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

	if num_threads:
		return tf.Session(config=tf.ConfigProto(
			gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
	else:
		return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


KTF.set_session(get_session(gpu_fraction=0.35)) # NEW 28-9-2017

print(keras.__version__) # == 2.1.5

"""
Paradigm of GAN (keras implementation)

1. Construct D
	1a) Compile D
2. Construct G
3. Set D.trainable = False
4. Stack G and D, to construct GAN 
	 4a) Compile GAN

"""

class GAN_test(object):
	"""docstring for GAN_test"""
	def __init__(self):
		# super(GAN_test, self).__init__()

		# self.arg = arg
		self.discriminator = None
		self.generator = None
		self.combined = None
		self.message_trainable_weights_of_combined_model = None # after compile of combine model
		self.message_trainable_weights_of_discriminator_model = None # after compile of discriminator model

	def _build_discriminator(self):

		img = Input(shape=(28,28,1), name="image")
		out = Dense(1, activation="sigmoid")(Flatten()(img))

		return Model(inputs=img, outputs=out)

	def _build_generator(self):

		noise = Input(shape=(10,), name="noise")
		layer = Dense(784, activation="tanh")(noise) # output image with each pixel in range (-1,1)
		img = Reshape((28,28,1))(layer)

		return Model(inputs=noise, outputs=img)

	def build_gan(self):

		# img = Input(shape=(28,28,1), name="image")
		noise = Input(shape=(10,), name="noise")

		self.discriminator = self._build_discriminator()
		self.discriminator.name = "discriminator"
		self.discriminator.compile(loss="binary_crossentropy", optimizer='adam', metrics=["acc"])

		self.message_trainable_weights_of_discriminator_model = self.discriminator.trainable_weights
		print("="*50)
		print("It's not intuitive here:")
		print(len(self.discriminator.trainable_weights)) # stdout = 2
		print(self.discriminator.trainable_weights)
		self.discriminator.trainable = False
		# self.discriminator._collected_trainable_weights = [] # DO NOT use this !!! 

		print(len(self.discriminator.trainable_weights)) # stdout = 0 it's not intuitive here to have 0, but it's normal.
		print(len(self.discriminator._collected_trainable_weights)) # stdout = 2 # trainable weights during training
		print("="*50)


		self.generator = self._build_generator()
		self.generator.name = "generator"
		generated_img = self.generator([noise])
		print(len(self.generator.trainable_weights)) # stdout = 2

		output = self.discriminator([generated_img])
		self.combined = Model(inputs=[noise], outputs=[output])
		self.combined.compile(loss="binary_crossentropy", optimizer='adam', metrics=["acc"])
		

		self.message_trainable_weights_of_combined_model = self.combined.trainable_weights
		


	def summary(self):

		print("="*50)
		print("Discriminator summary")
		self.discriminator.summary()
		print("="*50)
		print("Generator summary")
		self.generator.summary()
		print("="*50)
		print("Combined summary")
		self.combined.summary()
		print()
		
		print()
		print("="*50)
		print("Trainable weights of D model (just after compiled 'D'):\n {}".format(self.message_trainable_weights_of_discriminator_model))
		print()
		print("Trainable weights of combined model (just after compiled 'combine'):\n {}".format(self.message_trainable_weights_of_combined_model))
		print("="*50)
		print()
	def verify_trainable_layers(self):

		print(len(self.discriminator.trainable_weights))
		print(len(self.generator.trainable_weights))
		print(len(self.combined.trainable_weights))
		print(self.generator.trainable_weights)
	

	def linear_reg(self):
		img = Input(shape=(28,28,1))

		self.regressor = self._build_discriminator()
		self.regressor.summary()
		self.regressor.trainable = False
		self.regressor.summary()
		print(self.regressor.trainable_weights)
		# print(self.regressor._collected_trainable_weights)

	def train(self, batch_size=32):


		for iteration in range(100):

			#====================
			# Train Discriminator
			#====================
			Images_real = np.random.uniform(-1,1, (batch_size, 28,28,1))
			noise = np.random.uniform(-1,1, (batch_size, 10))
			fake_img = self.generator.predict(noise)
			
			fake = np.zeros((batch_size, 1)) # Fake images
			real = np.ones((batch_size, 1)) # Real images

			D_loss_fake = self.discriminator.train_on_batch(fake_img, fake)
			D_loss_real = self.discriminator.train_on_batch(Images_real, real)
			D_loss = 0.5*np.add(D_loss_fake, D_loss_real)

			# Save number of trainable weights of D during training 
			D_trainable_weights = len(self.discriminator._collected_trainable_weights)

			print("Nbs of trainable weigths of D during training of D: {}".format(D_trainable_weights))
			G_trainable_weights = len(self.generator.trainable_weights)

			#=============================================
			# Train Generator while freezing Discriminator
			#=============================================
			
			####### To see effect of 'compile' #######
			# self.discriminator.trainable = True  # only this line is insufficient 
			# self.combined.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
			
			noise = np.random.uniform(-1,1, (batch_size, 10))
			make_img_real = np.ones((batch_size, 1)) # Making fake images looks real.

			save_D_weights_before_training_G = self.combined.get_layer("discriminator").get_layer("dense_1").get_weights()

			G_loss = self.combined.train_on_batch(noise, make_img_real)
			save_D_weights_after_training_G = self.combined.get_layer("discriminator").get_layer("dense_1").get_weights()


			print("Nbs of trainable weigths of D during training of G: {}".format(len(self.discriminator.trainable_weights)))
			print("Nbs of trainable weights of Combined(G+D) during training: {}".format(len(self.combined._collected_trainable_weights)))
			print("Discriminator's weights not changed after training of G: {}".format(np.allclose(save_D_weights_before_training_G[0], save_D_weights_after_training_G[0])))
			# Assert that Combined model would not affect D's trainable weights during training 
			assert D_trainable_weights == len(self.discriminator._collected_trainable_weights)
			# Assert that Discriminator is freezed during training of Combined model
			assert G_trainable_weights == len(self.combined._collected_trainable_weights)

			# Assert that trainable weights combined == G+D
			# assert len(self.combined._collected_trainable_weights) == G_trainable_weights + len(self.discriminator.trainable_weights)

			
			print("[D loss: {} acc: {}] [G loss: {} acc: {}]".format(D_loss[0], D_loss[1], G_loss[0], G_loss[1]))



if __name__=="__main__":
	print("Start")

	gan = GAN_test()
	# gan.linear_reg()
	gan.build_gan()
	gan.summary()
	gan.train()
	# gan.verify_trainable_layers()