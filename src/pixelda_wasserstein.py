from __future__ import print_function, division
import scipy
import os
# import keras
# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", help="Choose the mode of GPU.", type=str, default="simple", choices=["simple", "multi"])
parser.add_argument("-v", help="verbose.", default=0, type=int, choices=[0, 1])
args = parser.parse_args()

if args.gpu == "simple":
	# simple GPU
	
	import socket
	machine_name = socket.gethostname()
	print("="*50)
	print("Machine name: {}".format(machine_name))
	print("="*50)
	if machine_name == "lulin-QX-350-Series":
		os.environ["CUDA_VISIBLE_DEVICES"]="0"
	else:
		os.environ["CUDA_VISIBLE_DEVICES"]="1"
		import matplotlib as mpl 
		mpl.use("Agg")
		# Qt_XKB_CONFIG_ROOT (add path ?)

	import keras
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

	if machine_name == "lulin-QX-350-Series":
		print("Using local machine..")
		KTF.set_session(get_session(gpu_fraction=0.4)) # NEW 28-9-2017
	else:
		KTF.set_session(get_session(gpu_fraction=0.95)) # NEW 28-9-2017
		

elif args.gpu == "multi":
	import keras
	from keras.utils import multi_gpu_model
else:
	print(args.gpu)





from keras.datasets import mnist
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model, Sequential #load_model
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils import to_categorical
import datetime
import matplotlib.pyplot as plt
import sys
from data_processing import DataLoader
import numpy as np
from keras.layers.merge import _Merge
from keras import backend as K
from time import time
from functools import partial



def wasserstein_loss(y_true, y_pred):
	"""Calculates the Wasserstein loss for a sample batch.
	The Wasserstein loss function is very simple to calculate. In a standard GAN, the discriminator
	has a sigmoid output, representing the probability that samples are real or generated. In Wasserstein
	GANs, however, the output is linear with no activation function! Instead of being constrained to [0, 1],
	the discriminator wants to make the distance between its output for real and generated samples as large as possible.
	The most natural way to achieve this is to label generated samples -1 and real samples 1, instead of the
	0 and 1 used in normal GANs, so that multiplying the outputs by the labels will give you the loss immediately.
	Note that the nature of this loss means that it can be (and frequently will be) less than 0."""
	return -K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
	"""Calculates the gradient penalty loss for a batch of "averaged" samples.
	In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
	that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
	this function at all points in the input space. The compromise used in the paper is to choose random points
	on the lines between real and generated samples, and check the gradients at these points. Note that it is the
	gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
	In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
	Then we get the gradients of the discriminator w.r.t. the input averaged samples.
	The l2 norm and penalty can then be calculated for this gradient.
	Note that this loss function requires the original averaged samples as input, but Keras only supports passing
	y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
	averaged_samples argument, and use that for model training."""
	# first get the gradients:
	#   assuming: - that y_pred has dimensions (batch_size, 1)
	#             - averaged_samples has dimensions (batch_size, nbr_features)
	# gradients afterwards has dimension (batch_size, nbr_features), basically
	# a list of nbr_features-dimensional gradient vectors
	gradients = K.gradients(y_pred, averaged_samples)[0]
	# compute the euclidean norm by squaring ...
	gradients_sqr = K.square(gradients)
	#   ... summing over the rows ...
	gradients_sqr_sum = K.sum(gradients_sqr,
							  axis=np.arange(1, len(gradients_sqr.shape)))
	#   ... and sqrt
	gradient_l2_norm = K.sqrt(gradients_sqr_sum)
	# compute lambda * (1 - ||grad||)^2 still for each single sample
	gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
	# return the mean as loss over all the batch samples
	return K.mean(gradient_penalty)

class RandomWeightedAverage(_Merge):
	"""Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
	between each pair of input points.
	Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
	Improvements appreciated."""
	def __init__(self, batch_size=32):
		super(RandomWeightedAverage, self).__init__()
		self.batch_size = batch_size

	def _merge_function(self, inputs):
		weights = K.random_uniform((1, 1, 1))
		return (weights * inputs[0]) + ((1 - weights) * inputs[1])


		

class PixelDA(object):
	"""
	Paradigm of GAN (keras implementation)

	1. Construct D
		1a) Compile D
	2. Construct G
	3. Set D.trainable = False
	4. Stack G and D, to construct GAN (combined model)
		 4a) Compile GAN
	
	Approved by fchollet: "the process you describe is in fact correct."

	See issue #4674 keras: https://github.com/keras-team/keras/issues/4674
	"""
	def __init__(self, noise_size=100, use_PatchGAN=False, use_Wasserstein=True):
		# Input shape
		self.img_rows = 32
		self.img_cols = 32
		self.channels = 3
		self.img_shape = (self.img_rows, self.img_cols, self.channels)
		self.num_classes = 10
		self.noise_size = (noise_size,)#(100,)

		# Calculate output shape of D (PatchGAN)
		patch = int(self.img_rows / 2**4)
		self.disc_patch = (patch, patch, 1)

		# Number of residual blocks in the generator
		self.residual_blocks = 6
		self.use_PatchGAN = use_PatchGAN #False
		self.use_Wasserstein = use_Wasserstein

		if self.use_Wasserstein:
			self.critic_steps = 5
		else:
			self.critic_steps = 1
		
		self.GRADIENT_PENALTY_WEIGHT = 10  # As the paper

	def build_all_model(self, batch_size=32):
		self.batch_size = batch_size
		# Loss weights
		lambda_adv = 10
		lambda_clf = 1
		# optimizer = Adam(0.0002, 0.5)
		# optimizer = SGD(lr=0.0001)
		optimizer = RMSprop(lr=1e-5)

		# Number of filters in first layer of discriminator and classifier
		self.df = 64
		self.cf = 64

		# Build and compile the discriminators
		self.discriminator = self.build_discriminator()
		self.discriminator.name = "Discriminator"


		img_A = Input(shape=self.img_shape, name='source_image') # real A
		img_B = Input(shape=self.img_shape, name='target_image') # real B
		fake_img = Input(shape=self.img_shape, name="fake_image") # fake B

		# We also need to generate weighted-averages of real and generated samples, to use for the gradient norm penalty.
		avg_img = RandomWeightedAverage(batch_size=self.batch_size)([img_B, fake_img])
		

		real_img_rating = self.discriminator(img_B) # TODO img_A
		fake_img_rating = self.discriminator(fake_img)
		avg_img_output = self.discriminator(avg_img)

		# The gradient penalty loss function requires the input averaged samples to get gradients. However,
		# Keras loss functions can only have two arguments, y_true and y_pred. We get around this by making a partial()
		# of the function with the averaged samples here.
		partial_gp_loss = partial(gradient_penalty_loss,
					  averaged_samples=avg_img,
					  gradient_penalty_weight=self.GRADIENT_PENALTY_WEIGHT)
		partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error

		if self.use_Wasserstein:
			self.discriminator_model = Model(inputs=[img_B, fake_img],  #, avg_img
											# loss_weights=[1,1,1], # useless, since we have multiply the penalization by GRADIENT_PENALTY_WEIGHT=10
											outputs=[real_img_rating, fake_img_rating, avg_img_output])
			self.discriminator_model.compile(loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],
				optimizer=optimizer,
				metrics=['accuracy'])
		else:
			self.discriminator.compile(loss='mse',
				optimizer=optimizer,
				metrics=['accuracy'])



		# For the combined model we will only train the generator and classifier
		self.discriminator.trainable = False

		# Build the generator
		self.generator = self.build_generator()
		self.generator.name = "Generator"
		# Build the task (classification) network
		self.clf = self.build_classifier()
		self.clf.name = "Classifier" 
		# Input images from both domains

		
		# Input noise
		noise = Input(shape=self.noise_size, name='noise_input')

		# Translate images from domain A to domain B
		fake_B = self.generator([img_A, noise])

		# Classify the translated image
		class_pred = self.clf(fake_B)

		
		# Discriminator determines validity of translated images
		valid = self.discriminator(fake_B) # fake_B_rating
		if self.use_Wasserstein:
			self.combined = Model(inputs=[img_A, noise], outputs=[valid, class_pred])
			self.combined.compile(optimizer=optimizer, 
									loss=[wasserstein_loss, 'categorical_crossentropy'],
									loss_weights=[lambda_adv, lambda_clf], # TODO NEW 
									metrics=['accuracy'])
		else:
			self.combined = Model([img_A, noise], [valid, class_pred])
			self.combined.compile(loss=['mse', 'categorical_crossentropy'],
										loss_weights=[lambda_adv, lambda_clf],
										optimizer=optimizer,
										metrics=['accuracy'])



	def load_dataset(self):
		# Configure MNIST and MNIST-M data loader
		self.data_loader = DataLoader(img_res=(self.img_rows, self.img_cols))

	def build_generator(self):
		"""Resnet Generator"""

		def residual_block(layer_input):
			"""Residual block described in paper"""
			d = Conv2D(64, kernel_size=3, strides=1, padding='same')(layer_input)
			d = BatchNormalization(momentum=0.8)(d) # TODO 6/5/2018
			d = Activation('relu')(d)
			d = Conv2D(64, kernel_size=3, strides=1, padding='same')(d)
			d = BatchNormalization(momentum=0.8)(d) # TODO 6/5/2018
			d = Add()([d, layer_input])
			return d

		# Image input
		img = Input(shape=self.img_shape, name='image_input')

		## Noise input
		noise = Input(shape=self.noise_size, name='noise_input')
		noise_layer = Dense(1024, activation="relu")(noise)
		noise_layer = Reshape((self.img_rows,self.img_cols, 1))(noise_layer)
		conditioned_img = keras.layers.concatenate([img, noise_layer])
		# keras.layers.concatenate

		# l1 = Conv2D(64, kernel_size=3, padding='same', activation='relu')(img)
		l1 = Conv2D(64, kernel_size=3, padding='same', activation='relu')(conditioned_img)
		

		# Propogate signal through residual blocks
		r = residual_block(l1)
		for _ in range(self.residual_blocks - 1):
			r = residual_block(r)

		output_img = Conv2D(self.channels, kernel_size=3, padding='same', activation='tanh')(r)

		return Model([img, noise], output_img)


	# def build_discriminator(self):

	# 	model = Sequential()
	# 	model.add(Input(shape=self.img_shape, name='image'))
	# 	def d_layer(model, filters, f_size=4, normalization=True):
	# 		"""Discriminator layer"""
	# 		model.add(Conv2D(filters, kernel_size=f_size, strides=2, padding='same'))
	# 		model.add(LeakyReLU(alpha=0.2))

	# 		if normalization:
	# 			model.add(InstanceNormalization())
	# 		return model

	# 	model = d_layer(model, self.df, normalization=False)
	# 	model = d_layer(model, self.df*2)
	# 	model = d_layer(model, self.df*4)
	# 	model = d_layer(model, self.df*8)

	# 	if self.use_PatchGAN: # NEW 7/5/2018
	# 		model.add(Conv2D(1, kernel_size=4, strides=1, padding='same'))
	# 	else:
	# 		if self.use_Wasserstein: # NEW 8/5/2018
	# 			model.add(Flatten())
	# 			model.add(Dense(1, kernel_initializer='he_normal')) # he_normal ?? TODO
	# 		else:
	# 			model.add(Flatten())
	# 			model.add(Dense(1, activation='sigmoid'))
	# 	return model
	def build_discriminator(self):

		def d_layer(layer_input, filters, f_size=4, normalization=True):
			"""Discriminator layer"""
			d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
			d = LeakyReLU(alpha=0.2)(d)
			if normalization:
				d = InstanceNormalization()(d)
			return d

		img = Input(shape=self.img_shape, name="image")

		d1 = d_layer(img, self.df, normalization=False)
		d2 = d_layer(d1, self.df*2, normalization=False)
		d3 = d_layer(d2, self.df*4, normalization=False)
		d4 = d_layer(d3, self.df*8, normalization=False)

		if self.use_PatchGAN: # NEW 7/5/2018
			validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
		else:
			if self.use_Wasserstein: # NEW 8/5/2018
				validity = Dense(1, kernel_initializer='he_normal')(Flatten()(d4)) # he_normal ?? TODO
			else:
				validity = Dense(1, activation='sigmoid')(Flatten()(d4))
			

		return Model(img, validity)

	def build_classifier(self):

		def clf_layer(layer_input, filters, f_size=4, normalization=True):
			"""Classifier layer"""
			d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
			d = LeakyReLU(alpha=0.2)(d)
			if normalization:
				d = InstanceNormalization()(d)
			return d

		img = Input(shape=self.img_shape, name='image_input')

		c1 = clf_layer(img, self.cf, normalization=False)
		c2 = clf_layer(c1, self.cf*2)
		c3 = clf_layer(c2, self.cf*4)
		c4 = clf_layer(c3, self.cf*8)
		c5 = clf_layer(c4, self.cf*8)

		class_pred = Dense(self.num_classes, activation='softmax')(Flatten()(c5))

		return Model(img, class_pred)
	def load_pretrained_weights(self, weights_path="../Weights/all_weights.h5"):
		print("Loading pretrained weights from path: {} ...".format(weights_path))

		self.combined.load_weights(weights_path, by_name=True)
		print("+ Done.")
	def summary(self):
		print("="*50)
		print("Discriminator summary:")
		self.discriminator.summary()
		print("="*50)
		print("Generator summary:")
		self.generator.summary()
		print("="*50)
		print("Classifier summary:")
		self.clf.summary()
		
		if self.use_Wasserstein:
			print("="*50)
			print("Discriminator model summary:")
			self.discriminator_model.summary()
		print("="*50)
		print("Combined model summary:")
		self.combined.summary()


	def train(self, epochs, batch_size=32, sample_interval=50, save_sample2dir="../samples/exp0", save_weights_path='../Weights/all_weights.h5', save_model=False):
		dirpath = "/".join(save_weights_path.split("/")[:-1])
		if not os.path.exists(dirpath):
			os.makedirs(dirpath)

		# half_batch = batch_size #int(batch_size / 2) ### TODO
		half_batch = int(batch_size / 2)
		
		# Classification accuracy on 100 last batches of domain B
		test_accs = []


		## Monitor to save model weights Lu
		best_test_cls_acc = 0
		second_best_cls_acc = -1
		for epoch in range(epochs):

			# ---------------------
			#  Train Discriminator
			# ---------------------
			# n_sample = half_batch # imgs_A.shape[0]
			
			for _ in range(self.critic_steps):

				imgs_A, _ = self.data_loader.load_data(domain="A", batch_size=half_batch)
				imgs_B, _ = self.data_loader.load_data(domain="B", batch_size=half_batch)
				
				
				noise_prior = np.random.normal(0,1, (half_batch, self.noise_size[0])) # TODO
				# noise_prior = np.random.rand(half_batch, self.noise_size[0]) # TODO 6/5/2018
				
				# Translate images from domain A to domain B
				fake_B = self.generator.predict([imgs_A, noise_prior])
				if self.use_PatchGAN:
					valid = np.ones((half_batch,) + self.disc_patch)
					fake = np.zeros((half_batch,) + self.disc_patch)
				else:
					if self.use_Wasserstein:
						valid = np.ones((half_batch, 1))
						fake = - np.ones((half_batch, 1)) # = - valid ? TODO
						dummy_y = np.zeros((batch_size, 1)) # NEW
					else:
						valid = np.ones((half_batch, 1))
						fake = np.zeros((half_batch, 1))
				# fake = -valid # TODO 6/5/2018 NEW
				D_train_images = np.vstack([imgs_B, fake_B]) # 6/5/2018 NEW
				D_train_label = np.vstack([valid, fake]) # 6/5/2018 NEW
				

				# Train the discriminators (original images = real / translated = Fake)
				# d_loss_real = self.discriminator.train_on_batch(imgs_B, valid)
				# d_loss_fake = self.discriminator.train_on_batch(fake_B, fake)
				# d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
				if self.use_Wasserstein:
					d_loss = self.discriminator_model.train_on_batch(D_train_images, D_train_label, dummy_y)
					# d_loss = self.discriminator.train_on_batch(D_train_images, D_train_label, dummy_y)
				else:
					d_loss = self.discriminator.train_on_batch(D_train_images, D_train_label)


			# --------------------------------
			#  Train Generator and Classifier
			# --------------------------------

			# Sample a batch of images from both domains
			imgs_A, labels_A = self.data_loader.load_data(domain="A", batch_size=batch_size)
			imgs_B, labels_B = self.data_loader.load_data(domain="B", batch_size=batch_size)

			# One-hot encoding of labels
			labels_A = to_categorical(labels_A, num_classes=self.num_classes)

			# The generators want the discriminators to label the translated images as real
			if self.use_PatchGAN:
				valid = np.ones((batch_size,) + self.disc_patch)
			else:
				valid = np.ones((batch_size, 1))

			#
			noise_prior = np.random.normal(0,1, (batch_size, self.noise_size[0])) # TODO
			# noise_prior = np.random.rand(batch_size, self.noise_size[0]) # TODO 6/5/2018

			# Train the generator and classifier
			g_loss = self.combined.train_on_batch([imgs_A, noise_prior], [valid, labels_A])

			#-----------------------
			# Evaluation (domain B)
			#-----------------------

			pred_B = self.clf.predict(imgs_B)
			test_acc = np.mean(np.argmax(pred_B, axis=1) == labels_B)

			# Add accuracy to list of last 100 accuracy measurements
			test_accs.append(test_acc)
			if len(test_accs) > 100:
				test_accs.pop(0)


			# Plot the progress
			# print ( "%d : [D - loss: %.5f, acc: %3d%%], [G - loss: %.5f], [clf - loss: %.5f, acc: %3d%%, test_acc: %3d%% (%3d%%)]" % \
			# 								(epoch, d_loss[0], 100*float(d_loss[1]),
			# 								g_loss[1], g_loss[2], 100*float(g_loss[-1]),
			# 								100*float(test_acc), 100*float(np.mean(test_accs))))
			
			if epoch % 10 == 0:
				
				
				d_train_acc = 100*float(d_loss[1])
				
				gen_loss = g_loss[1]

				clf_train_acc = 100*float(g_loss[-1])
				clf_train_loss = g_loss[2]

				current_test_acc = 100*float(test_acc)
				test_mean_acc = 100*float(np.mean(test_accs))

				

				if test_mean_acc > best_test_cls_acc:
					second_best_cls_acc = best_test_cls_acc
					best_test_cls_acc = test_mean_acc
					
					if save_model:
						self.combined.save(save_weights_path)
					else:
						self.combined.save_weights(save_weights_path)	
					print("{} : [D - loss: {:.5f}, acc: {:.2f}%], [G - loss: {:.5f}], [clf - loss: {:.5f}, acc: {:.2f}%, test_acc: {:.2f}% ({:.2f}%)] (latest)".format(epoch, d_loss[0], d_train_acc, gen_loss, clf_train_loss, clf_train_acc, current_test_acc, test_mean_acc))
				elif test_mean_acc > second_best_cls_acc:
					second_best_cls_acc = test_mean_acc
					
					if save_model:
						self.combined.save(save_weights_path)
					else:
						self.combined.save_weights(save_weights_path[:-3]+"_bis.h5")
					print("{} : [D - loss: {:.5f}, acc: {:.2f}%], [G - loss: {:.5f}], [clf - loss: {:.5f}, acc: {:.2f}%, test_acc: {:.2f}% ({:.2f}%)] (before latest)".format(epoch, d_loss[0], d_train_acc, gen_loss, clf_train_loss, clf_train_acc, current_test_acc, test_mean_acc))

				else:

					print("{} : [D - loss: {:.5f}, acc: {:.2f}%], [G - loss: {:.5f}], [clf - loss: {:.5f}, acc: {:.2f}%, test_acc: {:.2f}% ({:.2f}%)]".format(epoch, d_loss[0], d_train_acc, gen_loss, clf_train_loss, clf_train_acc, current_test_acc, test_mean_acc))


			# If at save interval => save generated image samples
			if epoch % sample_interval == 0:
				self.sample_images(epoch, save2dir=save_sample2dir)
			
				

	def sample_images(self, epoch, save2dir="../samples"):
		if not os.path.exists(save2dir):
			os.makedirs(save2dir)

		r, c = 2, 10

		imgs_A, _ = self.data_loader.load_data(domain="A", batch_size=c)

		n_sample = imgs_A.shape[0]
		noise_prior = np.random.normal(0,1, (n_sample, self.noise_size[0])) # TODO
		# noise_prior = np.random.rand(n_sample, self.noise_size[0]) # TODO 6/5/2018

		# Translate images to the other domain
		fake_B = self.generator.predict([imgs_A, noise_prior])

		gen_imgs = np.concatenate([imgs_A, fake_B])

		# Rescale images 0 - 1
		gen_imgs = 0.5 * gen_imgs + 0.5

		#titles = ['Original', 'Translated']
		fig, axs = plt.subplots(r, c, figsize=(20, 4))
		cnt = 0
		for i in range(r):
			for j in range(c):
				axs[i,j].imshow(gen_imgs[cnt])
				#axs[i, j].set_title(titles[i])
				axs[i,j].axis('off')
				cnt += 1
		fig.savefig(os.path.join(save2dir, "{}.png".format(epoch)))
		plt.close()


	def deploy_transform(self, save2file="../domain_adapted/generated.npy", stop_after=None):
		dirpath = "/".join(save2file.split("/")[:-1])
		if not os.path.exists(dirpath):
			os.makedirs(dirpath)

		dirname = "/".join(save2file.split("/")[:-1])

		
		if stop_after is not None:
			predict_steps = int(stop_after/32)
		else:
			predict_steps = stop_after

		noise_vec = np.random.normal(0,1, self.noise_size[0])
		assert 1==2
		# np.random.rand(n_sample, self.noise_size[0]) # TODO 6/5/2018

		print("Performing Pixel-level domain adaptation on original images...")
		adaptaed_images = self.generator.predict([self.data_loader.mnist_X[:32*predict_steps], np.tile(noise_vec, (32*predict_steps,1))], batch_size=32) #, steps=predict_steps
		# self.data_loader.mnistm_X[:stop_after]
		print("+ Done.")
		print("Saving transformed images to file {}".format(save2file))
		np.save(save2file, adaptaed_images)

		noise_vec_filepath = os.path.join(dirname, "noise_vectors.npy")
		print("Saving random noise (seed) to file {}".format(noise_vec_filepath))
		np.save(noise_vec_filepath, noise_vec)

		print("+ All done.")

	def deploy_debug(self, save2file="../domain_adapted/debug.npy", sample_size=9, seed = 0):
		dirpath = "/".join(save2file.split("/")[:-1])
		if not os.path.exists(dirpath):
			os.makedirs(dirpath)

		dirname = "/".join(save2file.split("/")[:-1])

		np.random.seed(seed=seed)

		noise_vec = np.random.normal(0,1, (sample_size, self.noise_size[0]))
		assert 1==2
		# np.random.rand(n_sample, self.noise_size[0]) # TODO 6/5/2018


		print("Performing Pixel-level domain adaptation on original images...")
		collections = []
		for i in range(sample_size):
			adaptaed_images = self.generator.predict([self.data_loader.mnist_X[:15], np.tile(noise_vec[i], (15,1))], batch_size=15)

			collections.append(adaptaed_images)
		print("+ Done.")

		print("Saving transformed images to file {}".format(save2file))
		np.save(save2file, np.stack(collections))
		print("+ All done.")
	
	def deploy_classification(self, batch_size=32):
		print("Predicting ... ")
		pred_B = self.clf.predict(self.data_loader.mnistm_X, batch_size=batch_size)
		print("+ Done.")
		N_samples = len(pred_B)
		precision = (np.argmax(pred_B, axis=1) == self.data_loader.mnistm_y)
		Moy = np.mean(precision)
		Std = np.std(precision)

		lower_bound = Moy - 2.576*Std/np.sqrt(N_samples) 
		upper_bound = Moy + 2.576*Std/np.sqrt(N_samples)
		print("="*50)
		print("Unsupervised MNIST-M classification accuracy : {}".format(Moy))
		print("Confidence interval (99%) [{}, {}]".format(lower_bound, upper_bound))
		print("Length of confidence interval 99%: {}".format(upper_bound-lower_bound))
		print("="*50)
		print("+ All done.")

if __name__ == '__main__':
	gan = PixelDA(noise_size=100, use_PatchGAN=False, use_Wasserstein=True)
	gan.build_all_model()
	# gan.load_dataset()
	gan.summary()

	# gan.load_pretrained_weights(weights_path='../Weights/exp6.h5')
	# gan.train(epochs=2000, batch_size=32, sample_interval=100)
	# gan.train(epochs=40000, batch_size=32, sample_interval=100, save_sample2dir="../samples/exp9", save_weights_path='../Weights/exp9.h5')
	# gan.train(epochs=10000, batch_size=32, sample_interval=100, save_sample2dir="../samples/Exp0_no_batchnorm/exp0", save_weights_path='../Weights/Exp0_no_batchnorm/exp0.h5', save_model=False)
	# gan.train(epochs=20000, batch_size=32, sample_interval=100, save_sample2dir="../samples/Exp0_rand_noise_100/exp0", save_weights_path='../Weights/Exp0_rand_noise_100/exp0.h5', save_model=False)
	# gan.train(epochs=20000, batch_size=32, sample_interval=100, save_sample2dir="../samples/Exp0_gaussian_noise_100_no_batchnorm/exp0", save_weights_path='../Weights/Exp0_gaussian_noise_100_no_batchnorm/exp0.h5', save_model=False)
	# gan.deploy_transform(stop_after=200)
	# gan.deploy_transform(stop_after=400, save2file="../domain_adapted/Exp7/generated.npy")
	# gan.deploy_debug(save2file="../domain_adapted/Exp7/debug.npy", sample_size=100, seed = 0)
	# gan.deploy_classification()