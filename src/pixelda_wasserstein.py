from __future__ import print_function, division
import scipy
import os, sys
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
		sys.path.append("/home/lulin/Desktop/Desktop/Python_projets/my_packages")
	else:
		os.environ["CUDA_VISIBLE_DEVICES"]="1"
		sys.path.append("/home/lulin/na4/my_packages")

		import matplotlib as mpl 
		# mpl.use("Agg")
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
		KTF.set_session(get_session(gpu_fraction=0.35)) # NEW 28-9-2017
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

from data_processing import DataLoader
import numpy as np
from keras.layers.merge import _Merge
from keras import backend as K
from time import time
from functools import partial
from keras.utils import plot_model

try:
	from HPOlib_lu.Quasi_Monte_Carlo.sobol_lib import i4_sobol_generate
except:
	print("Can't import Sobol library.")
	pass

from tqdm import tqdm
import dill
from DLalgors import _DLalgo
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
	def __init__(self):
		super(RandomWeightedAverage, self).__init__()


	def _merge_function(self, inputs):
		weights = K.random_uniform((1, 1, 1))
		return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def my_critic_acc(y_true, y_pred):
	sign = K.less(K.zeros(1), y_true*y_pred)
	return K.mean(sign)

class PixelDA(_DLalgo):
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
	def __init__(self, noise_size=(100,), 
		use_PatchGAN=False, 
		use_Wasserstein=True, 
		batch_size=64,
		**kwargs):
		# Input shape
		self.dataset_name = "MNIST" # "CT"

		if self.dataset_name == "MNIST":
			self.img_shape = (32, 32, 3)
			self.num_classes = 10
		elif self.dataset_name == "CT":
			self.img_shape = (64, 64, 1)
		else:
			assert 1==2, "Only support two datasets for now. ('CT', 'MNIST')"

		self.img_rows, self.img_cols, self.channels = self.img_shape

		self.noise_size = noise_size #(100,)
		self.batch_size = batch_size
		# Loss weights
		self.lambda_adv = 7#10 # 7
		self.lambda_clf = 1
		# Number of filters in first layer of discriminator and classifier
		self.df = 64 # NEW TODO #64 11/5/2018
		self.cf = 64

		self.normalize_G = False
		self.normalize_D = False
		self.normalize_C = True
		self.shift_label = False # TODO NEW 31/5/2018
		# Number of residual blocks in the generator
		self.residual_blocks = 17 # 6 # NEW TODO 14/5/2018
		self.use_PatchGAN = use_PatchGAN #False
		self.use_Wasserstein = use_Wasserstein
		if self.use_PatchGAN:
			# Calculate output shape of D (PatchGAN)
			patch = int(self.img_rows / 2**4)
			self.disc_patch = (patch, patch, 1)

		if self.use_Wasserstein:
			self.critic_steps = 5#5 #7 #10
		else:
			self.critic_steps = 1
		
		self.GRADIENT_PENALTY_WEIGHT = 10#10#5 #10 As the paper


		##### Set up the other attributes
		for key in kwargs:
			setattr(self, key, kwargs[key])

	def freeze_layers_kernel(self, model):
		valid_name = ["dense", "conv2d"]
		layers_names = list(map(lambda layer:layer.name, model.layers))
		num_layers = len(layers_names)
		valid_index = list(map(lambda layer_name:(layer_name.split('_')[0] in valid_name), layers_names))
		valid_index = np.arange(num_layers)[valid_index]

		for i in valid_index:
			## Remove 'kernel' from trainable weights before compile model !
			model.layers[i].trainable_weights = model.layers[i].trainable_weights[1:]

	def build_all_model(self):

		# optimizer = Adam(0.0002, 0.5)
		optimizer = Adam(0.0001, beta_1=0.5, beta_2=0.9)
		# optimizer = SGD(lr=0.0001)
		# optimizer = RMSprop(lr=1e-5)

		

		# Build and compile the discriminators
		self.discriminator = self.build_discriminator()
		self.discriminator.name = "Discriminator"


		img_A = Input(shape=self.img_shape, name='source_image') # real A
		img_B = Input(shape=self.img_shape, name='target_image') # real B
		fake_img = Input(shape=self.img_shape, name="fake_image") # fake B

		# We also need to generate weighted-averages of real and generated samples, to use for the gradient norm penalty.
		avg_img = RandomWeightedAverage()([img_B, fake_img])
		

		real_img_rating = self.discriminator(img_B) 
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

			### One time experimence: Freeze all Discriminator's 'kernel'
			# self.freeze_layers_kernel(self.discriminator) # TODO

			self.discriminator_model = Model(inputs=[img_B, fake_img],  #, avg_img
											# loss_weights=[1,1,1], # useless, since we have multiply the penalization by GRADIENT_PENALTY_WEIGHT=10
											outputs=[real_img_rating, fake_img_rating, avg_img_output])
			self.discriminator_model.compile(loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],
				optimizer=optimizer,
				metrics=[my_critic_acc])
			# print(len(self.discriminator_model._collected_trainable_weights))
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
									loss_weights=[self.lambda_adv, self.lambda_clf], 
									metrics=['accuracy'])
		else:
			self.combined = Model([img_A, noise], [valid, class_pred])
			self.combined.compile(loss=['mse', 'categorical_crossentropy'],
										loss_weights=[self.lambda_adv, self.lambda_clf],
										optimizer=optimizer,
										metrics=['accuracy'])



	def load_dataset(self):
		# Configure MNIST and MNIST-M data loader
		self.data_loader = DataLoader(img_res=(self.img_rows, self.img_cols))

	def build_generator(self):
		"""Resnet Generator"""

		def residual_block(layer_input, normalization=self.normalize_G):
			"""Residual block described in paper"""
			d = Conv2D(64, kernel_size=3, strides=1, padding='same')(layer_input)
			if normalization:
				d = InstanceNormalization()(d)
				# d = BatchNormalization(momentum=0.8)(d) # TODO 6/5/2018
			d = Activation('relu')(d)
			d = Conv2D(64, kernel_size=3, strides=1, padding='same')(d)
			if normalization:
				d = InstanceNormalization()(d)
				# d = BatchNormalization(momentum=0.8)(d) # TODO 6/5/2018
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


	def build_discriminator(self):

		def d_layer(layer_input, filters, f_size=4, normalization=self.normalize_D):
			"""Discriminator layer"""
			d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
			d = LeakyReLU(alpha=0.2)(d)
			if normalization:
				d = InstanceNormalization()(d)
			return d

		img = Input(shape=self.img_shape, name="image")

		d1 = d_layer(img, self.df, normalization=False)
		d2 = d_layer(d1, self.df*2, normalization=self.normalize_D)
		d3 = d_layer(d2, self.df*4, normalization=self.normalize_D)
		d4 = d_layer(d3, self.df*8, normalization=self.normalize_D)

		if self.use_PatchGAN: # NEW 7/5/2018
			validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
		else:
			if self.use_Wasserstein: # NEW 8/5/2018
				validity = Dense(1, activation=None)(Flatten()(d4)) # he_normal ?? TODO
			else:
				validity = Dense(1, activation='sigmoid')(Flatten()(d4))
			

		return Model(img, validity)

	def build_classifier(self):

		def clf_layer(layer_input, filters, f_size=4, normalization=self.normalize_C):
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
	def load_pretrained_weights(self, weights_path="../Weights/all_weights.h5", only_cls=False):
		print("Loading pretrained weights from path: {} ...".format(weights_path))
		if only_cls:
			# self.clf.load_weights(weights_path, by_name=True) # Don't work !
			# See: https://github.com/keras-team/keras/issues/5348
			# Solution:
			# Build a model, load (all) weights, save sub model weigths as np.array, kill(clear session) model
			# than finally, build new model, set sub model weights from pre-saved np.array !
			self.combined.load_weights(weights_path, by_name=True)
			clf_weights = self.clf.get_weights()
			K.clear_session()
			self.build_all_model()
			self.clf.set_weights(clf_weights)

		else:
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

	def write_tensorboard_graph(self, to_dir="../logs", save_png2dir="../Model_graph"):
		if not os.path.exists(save_png2dir):
			os.makedirs(save_png2dir)
		tensorboard = keras.callbacks.TensorBoard(log_dir=to_dir, histogram_freq=0, write_graph=True, write_images=False)
		# tensorboard.set_model(self.combined)
		tensorboard.set_model(self.discriminator_model)
		try:
			plot_model(self.combined, to_file=os.path.join(save_png2dir, "Combined_model.png"))
			plot_model(self.discriminator_model, to_file=os.path.join(save_png2dir, "Discriminator_model.png"))
		except:
			pass
		
	def train(self, epochs, sample_interval=50, save_sample2dir="../samples/exp0", save_weights_path='../Weights/all_weights.h5', save_model=False):
		dirpath = "/".join(save_weights_path.split("/")[:-1])
		if not os.path.exists(dirpath):
			os.makedirs(dirpath)
		self.save_config(save2path=os.path.join(dirpath, "config.dill"), verbose=True)

		#half_batch = batch_size #int(batch_size / 2) ### TODO
		# half_batch = int(batch_size / 2)
		
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

				imgs_A, _ = self.data_loader.load_data(domain="A", batch_size=self.batch_size)
				imgs_B, _ = self.data_loader.load_data(domain="B", batch_size=self.batch_size)
				
				
				noise_prior = np.random.normal(0,1, (self.batch_size, self.noise_size[0])) 
				# noise_prior = np.random.rand(half_batch, self.noise_size[0]) # TODO 6/5/2018
				
				# Translate images from domain A to domain B
				fake_B = self.generator.predict([imgs_A, noise_prior])
				if self.use_PatchGAN:
					valid = np.ones((self.batch_size,) + self.disc_patch)
					fake = np.zeros((self.batch_size,) + self.disc_patch)
				else:
					if self.use_Wasserstein:
						valid = np.ones((self.batch_size, 1))
						fake = - valid #np.ones((half_batch, 1)) # = - valid ? TODO
						dummy_y = np.zeros((self.batch_size, 1)) # NEW
					else:
						valid = np.ones((self.batch_size, 1))
						fake = np.zeros((self.batch_size, 1))
				
				
				

				# Train the discriminators (original images = real / translated = Fake)
				# d_loss_real = self.discriminator.train_on_batch(imgs_B, valid)
				# d_loss_fake = self.discriminator.train_on_batch(fake_B, fake)
				# d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
				if self.use_Wasserstein:
					d_loss = self.discriminator_model.train_on_batch([imgs_B, fake_B], [valid, fake, dummy_y])
					# d_loss = self.discriminator.train_on_batch(D_train_images, D_train_label, dummy_y)
				else:
					D_train_images = np.vstack([imgs_B, fake_B]) # 6/5/2018 NEW
					D_train_label = np.vstack([valid, fake]) # 6/5/2018 NEW
					d_loss = self.discriminator.train_on_batch(D_train_images, D_train_label)


			# --------------------------------
			#  Train Generator and Classifier
			# --------------------------------

			# Sample a batch of images from both domains
			imgs_A, labels_A = self.data_loader.load_data(domain="A", batch_size=self.batch_size)
			imgs_B, labels_B = self.data_loader.load_data(domain="B", batch_size=self.batch_size)
			if self.shift_label:
				labels_A = np.mod(labels_A+1, 10) #  == (labels_A+1) % 10

			# One-hot encoding of labels
			labels_A = to_categorical(labels_A, num_classes=self.num_classes)

			# The generators want the discriminators to label the translated images as real
			if self.use_PatchGAN:
				valid = np.ones((self.batch_size,) + self.disc_patch)
			else:
				valid = np.ones((self.batch_size, 1))

			#
			noise_prior = np.random.normal(0,1, (self.batch_size, self.noise_size[0])) 
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
				

				if self.use_Wasserstein:
					d_real_acc = 100*float(d_loss[4])
					d_fake_acc = 100*float(d_loss[5])
					d_train_acc = 100*(float(d_loss[4])+float(d_loss[5]))/2
				else:
					d_train_acc = 100*float(d_loss[1])
				
				gen_loss = g_loss[1]

				clf_train_acc = 100*float(g_loss[-1])
				clf_train_loss = g_loss[2]

				current_test_acc = 100*float(test_acc)
				test_mean_acc = 100*float(np.mean(test_accs))

				
				g_loss.append(current_test_acc)
				g_loss.append(test_mean_acc)

				with open(os.path.join(dirpath, "D_Losses.csv"), "ab") as csv_file:
					np.savetxt(csv_file, np.array(d_loss).reshape(1,-1), delimiter=",")
				with open(os.path.join(dirpath, "G_Losses.csv"), "ab") as csv_file:
					np.savetxt(csv_file, np.array(g_loss).reshape(1,-1), delimiter=",")

				message = "{} : [D - loss: {:.5f}, GP_loss: {:.5f}, (+) acc: {:.2f}%, (-) acc: {:.2f}%, acc: {:.2f}%], [G - loss: {:.5f}], [clf - loss: {:.5f}, acc: {:.2f}%, test_dice: {:.2f}% ({:.2f}%)]".format(epoch, d_loss[0], d_loss[3], d_real_acc, d_fake_acc, d_train_acc, gen_loss, clf_train_loss, clf_train_acc, current_test_acc, test_mean_acc)

				if test_mean_acc > best_test_cls_acc:
					second_best_cls_acc = best_test_cls_acc
					best_test_cls_acc = test_mean_acc
					
					if save_model:
						self.combined.save(save_weights_path)
					else:
						self.combined.save_weights(save_weights_path)
					message += "  (best)"
					 
				elif test_mean_acc > second_best_cls_acc:
					second_best_cls_acc = test_mean_acc
					
					if save_model:
						self.combined.save(save_weights_path)
					else:
						self.combined.save_weights(save_weights_path[:-3]+"_bis.h5")
					message += "  (second best)"

				else:
					pass
				print(message)


			# If at save interval => save generated image samples
			if epoch % sample_interval == 0:
				self.sample_images(epoch, save2dir=save_sample2dir)
			
				

	def sample_images(self, epoch, save2dir="../samples"):
		if not os.path.exists(save2dir):
			os.makedirs(save2dir)

		r, c = 5, 10

		imgs_A, _ = self.data_loader.load_data(domain="A", batch_size=c)

		n_sample = imgs_A.shape[0]

		gen_imgs = imgs_A
		for i in range(r-1):
			noise_prior = np.random.normal(0,1, (n_sample, self.noise_size[0])) # TODO
			# noise_prior = np.random.normal(0,3, (n_sample, self.noise_size[0])) # TODO # 16/5/2018
			# noise_prior = np.random.rand(n_sample, self.noise_size[0]) # TODO 6/5/2018

			# Translate images to the other domain
			fake_B = self.generator.predict([imgs_A, noise_prior])
			gen_imgs = np.concatenate([gen_imgs, fake_B])

		# Rescale images to 0 - 1
		gen_imgs = 0.5 * gen_imgs + 0.5

		#titles = ['Original', 'Translated']
		fig, axs = plt.subplots(r, c, figsize=(2*c, 2*r))

		cnt = 0
		for i in range(r):
			for j in range(c):
				axs[i,j].imshow(gen_imgs[cnt])
				#axs[i, j].set_title(titles[i])
				axs[i,j].axis('off')
				cnt += 1
		fig.savefig(os.path.join(save2dir, "{}.png".format(epoch)))
		plt.close()

	def train_segmenter(self, iterations, batch_size=32, noise_range=5, save_weights_path=None):
		if save_weights_path is not None:
			dirpath = "/".join(save_weights_path.split("/")[:-1])
			if not os.path.exists(dirpath):
				os.makedirs(dirpath)
		optimizer = Adam(0.000001, beta_1=0.0, beta_2=0.9)
		
		# Input noise
		noise = Input(shape=self.noise_size, name='noise_input_seg')
		img_A = Input(shape=self.img_shape, name='source_image_seg')
		# Translate images from domain A to domain B
		fake_B = self.generator([img_A, noise])

		# Segment the translated image
		mask_pred = self.seg(fake_B)

		self.generator.trainable = False

		self.segmentation_model = Model(inputs=[img_A, noise], outputs=[mask_pred])
		self.segmentation_model.compile(loss=dice_coef_loss, optimizer=optimizer, metrics=["acc", dice_coef])

		self.segmentation_model.name = "U-net (freeze Generator)"
		self.segmentation_model.summary()

		best_test_dice = 0.0
		second_best_test_dice = -1.0
		collections = []
		for e in range(iterations):
			noise = (2*np.random.random((batch_size, self.noise_size[0]))-1)*noise_range
			
			images_A, _, mask_A = self.data_loader.load_data(domain="A", batch_size=batch_size, return_mask=True)

			s_loss = self.segmentation_model.train_on_batch([images_A, noise], mask_A)


			if e%100 == 0:
				images_B, _, mask_B = self.data_loader.load_data(domain="B", batch_size=batch_size, return_mask=True)

				pred_mask_B = self.seg.predict(images_B)
				_, current_test_dice = dice_predict(mask_B, pred_mask_B)
				
				if len(collections)>=100:
					collections.pop(0)
				collections.append(current_test_dice)
				mean_dice = np.mean(collections)
				message = "{} dice loss: {:.3f}; acc: {:.5f}; mean dice (test): {:.3f}".format(e, 100*s_loss[0], s_loss[1], 100*mean_dice)

				if mean_dice>best_test_dice:
					best_test_dice = mean_dice
					message += "  (best)"
					if save_weights_path is not None:
						self.segmentation_model.save_weights(save_weights_path)

				elif mean_dice> second_best_test_dice:
					second_best_test_dice = mean_dice
					message += "  (second best)"
					if save_weights_path is not None:
						self.segmentation_model.save_weights(save_weights_path[:-3]+"_bis.h5")
				else:
					pass
				print(message)
				



		return


	def deploy_transform(self, save2file="../domain_adapted/generated.npy", noise_range=5.0, separate_cls=True, stop_after=None, seed=None):
		dirpath = "/".join(save2file.split("/")[:-1])
		if not os.path.exists(dirpath):
			os.makedirs(dirpath)

		dirname = "/".join(save2file.split("/")[:-1])
		
		# noise_vec = np.random.normal(0,1, self.noise_size[0])
		
		# np.random.rand(n_sample, self.noise_size[0]) # TODO 6/5/2018
		print("Performing Pixel-level domain adaptation on original images...")
		if separate_cls:
			for mnist_cls in tqdm(range(10)):

				imgs_A = self.data_loader.mnist_X[self.data_loader.mnist_y==mnist_cls]
				if stop_after is not None:
					num_samples = int(stop_after)
				else:
					num_samples = len(imgs_A)
				np.random.seed(seed)
				noise_vec = (2*np.random.random((num_samples, self.noise_size[0]))-1)*noise_range
				adaptaed_images = self.generator.predict([imgs_A[:num_samples], noise_vec], batch_size=32)
				filename_by_cls = save2file[:-4]+"_{}.npy".format(mnist_cls)
				np.save(filename_by_cls, adaptaed_images)
		else:

			imgs_A = self.data_loader.mnist_X
			if stop_after is not None:
					num_samples = int(stop_after)
			else:
				num_samples = len(imgs_A)

			np.random.seed(seed)
			noise_vec = (2*np.random.random((num_samples, self.noise_size[0]))-1)*noise_range
			adaptaed_images = self.generator.predict([imgs_A[:num_samples], noise_vec], batch_size=32) 
			print("Saving transformed images to file {}".format(save2file))
			np.save(save2file, adaptaed_images)
		
		# np.save(save2file, adaptaed_images)
		print("+ All done.")

	def deploy_debug(self, save2file="../domain_adapted/debug.npy", sample_size=100, noise_number=128, 
		use_sobol=False, 
		use_linear=False, 
		use_sphere=False, 
		use_uniform_linear=False, 
		use_zeros=False,
		seed = 17):
		dirpath = "/".join(save2file.split("/")[:-1])
		if not os.path.exists(dirpath):
			os.makedirs(dirpath)

		dirname = "/".join(save2file.split("/")[:-1])

		np.random.seed(seed=seed)
		
		# np.random.rand(n_sample, self.noise_size[0]) # TODO 6/5/2018

		print("Performing Pixel-level domain adaptation on original images...")
		# noise_vec = np.random.normal(0,1, (sample_size, self.noise_size[0]))
		# collections = []
		# for i in range(sample_size):
		# 	adaptaed_images = self.generator.predict([self.data_loader.mnist_X[:15], np.tile(noise_vec[i], (15,1))], batch_size=15)

		# 	collections.append(adaptaed_images)
		collections = []
		# imgs_A, _ = self.data_loader.load_data(domain="A", batch_size=sample_size)
		imgs_A = np.load("/home/lulin/na4/src/output/output13_32x32/test/liver_masks.npy")[100:100+sample_size]
		imgs_A = np.repeat(imgs_A, 3, axis=-1)

		for i in tqdm(range(sample_size)):
			if use_sobol:
				noise_vec = 5*(2*i4_sobol_generate(self.noise_size[0], noise_number, i*noise_number).T-1)
			elif use_linear:
				tangents = 3.0*(2*np.random.random((noise_number, 1))-1)
				noise_vec = np.ones((noise_number, self.noise_size[0]))*tangents
			elif use_sphere:
				noise_vec = 2*(np.random.random((noise_number, self.noise_size[0]))-1)
				norm_vec = np.linalg.norm(noise_vec, axis=-1)
				noise_vec = noise_vec/ norm_vec[:, np.newaxis]
			elif use_uniform_linear:
				tangents = 10.0*np.linspace(-1,1,noise_number)[:, np.newaxis]
				noise_vec = np.ones((noise_number, self.noise_size[0]))*tangents
			elif use_zeros:

				noise_vec = np.zeros((noise_number, self.noise_size[0]))
				
			else:
				noise_vec = np.random.normal(0,3, (noise_number, self.noise_size[0]))
			adaptaed_images = self.generator.predict([np.tile(imgs_A[i], (noise_number,1,1,1)), noise_vec], batch_size=32)
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

	def deploy_demo_only(self, save2file="../domain_adapted/WGAN_GP/Exp4/demo.npy", sample_size=25, noise_number=512, linspace_size=10.0):
		collections = []
		imgs_A, labels_A = self.data_loader.load_data(domain="A", batch_size=sample_size)


		tangents = linspace_size*np.linspace(-1,1,noise_number)[:, np.newaxis]
		noise_vec = np.ones((noise_number, self.noise_size[0]))*tangents

		for i in tqdm(range(noise_number)):
			adaptaed_images = self.generator.predict([imgs_A, np.tile(noise_vec[i],(sample_size, 1))], batch_size=sample_size)
			collections.append(adaptaed_images)
		print("+ Done.")

		print("Saving transformed images to file {}".format(save2file))
		np.save(save2file, np.stack(collections))
		print("+ All done.")

	def deploy_cherry_pick(self, save2file="../domain_adapted/WGAN_GP/Exp4/demo_cherry_picked.png", sample_size=25, noise_number=25, linspace_size=5.0):
		collections = []
		imgs_A, labels_A = self.data_loader.load_data(domain="A", batch_size=sample_size)
		assert noise_number == sample_size

		tangents = linspace_size*np.linspace(-1,1,noise_number)[:, np.newaxis]
		noise_vec = np.ones((noise_number, self.noise_size[0]))*tangents
		
		np.random.shuffle(noise_vec) # shuffle background color !

		
		adaptaed_images = self.generator.predict([imgs_A, noise_vec], batch_size=sample_size)
		adaptaed_images = (adaptaed_images+1)/2
		print("+ Done.")

		print("Saving transformed images to file {}".format(save2file))
		r = 5
		c = 5	
		fig, axs = plt.subplots(r, c, figsize=(5*c, 5*r))
		for j in range(r):
			for i in range(c):
				axs[j,i].imshow(adaptaed_images[c*j+i])
				axs[j,i].axis('off')
		plt.savefig(save2file)
		plt.close()
		print("+ All done.")


if __name__ == '__main__':
	# EXP_NAME = "Exp4_9"
	gan = PixelDA(noise_size=(100,), use_PatchGAN=False, use_Wasserstein=True, batch_size=64, shift_label=False)
	# gan.load_config(verbose=True, from_file="../Weights/WGAN_GP/{}/config.dill".format(EXP_NAME))
	gan.build_all_model()
	gan.summary()
	gan.load_dataset()
	
	### gan.save_config(verbose=True, save2path="../Weights/WGAN_GP/Exp4/config.dill")
	# gan.save_config(verbose=True, save2path="../Weights/WGAN_GP/Exp4_7/config.dill")
	gan.print_config()
	# gan.write_tensorboard_graph()
	# gan.load_pretrained_weights(weights_path='../Weights/WGAN_GP/Exp4_7/Exp0.h5')
	# gan.load_pretrained_weights(weights_path='../Weights/WGAN_GP/{}/Exp0.h5'.format(EXP_NAME), only_cls=False)



	# gan.deploy_debug(save2file="../domain_adapted/Others/Liver/mask.npy", sample_size=100, noise_number=256, seed = 17)
	# import ipdb; ipdb.set_trace()
	gan.train(epochs=100000, sample_interval=100, save_sample2dir="../samples/WGAN_GP/Exp_freeze", save_weights_path='../Weights/WGAN_GP/Exp_freeze/Exp0.h5')
	# gan.train(epochs=100000, batch_size=64, sample_interval=100, save_sample2dir="../samples/WGAN_GP/Exp3", save_weights_path='../Weights/WGAN_GP/Exp3/Exp3.h5')
	# gan.train(epochs=100000, batch_size=64, sample_interval=100, save_sample2dir="../samples/WGAN_GP/Exp4_13", save_weights_path='../Weights/WGAN_GP/Exp4_13/Exp0.h5')
	# gan.train(epochs=100000, batch_size=64, sample_interval=100, save_sample2dir="../samples/WGAN_GP/Exp4_14", save_weights_path='../Weights/WGAN_GP/Exp4_14/Exp0.h5')
	# gan.train(epochs=100000, batch_size=64, sample_interval=100, save_sample2dir="../samples/WGAN_GP/Exp4_14_1", save_weights_path='../Weights/WGAN_GP/Exp4_14_1/Exp0.h5')
	# gan.train(epochs=100000, batch_size=64, sample_interval=100, save_sample2dir="../samples/WGAN_GP/Exp5_1", save_weights_path='../Weights/WGAN_GP/Exp5_1/Exp0.h5')
	# gan.load_pretrained_weights(weights_path='../Weights/exp6.h5')
	# gan.train(epochs=2000, batch_size=32, sample_interval=100)
	# gan.train(epochs=40000, batch_size=32, sample_interval=100, save_sample2dir="../samples/exp9", save_weights_path='../Weights/exp9.h5')
	# gan.train(epochs=10000, batch_size=32, sample_interval=100, save_sample2dir="../samples/Exp0_no_batchnorm/exp0", save_weights_path='../Weights/Exp0_no_batchnorm/exp0.h5', save_model=False)
	# gan.train(epochs=20000, batch_size=32, sample_interval=100, save_sample2dir="../samples/Exp0_rand_noise_100/exp0", save_weights_path='../Weights/Exp0_rand_noise_100/exp0.h5', save_model=False)
	# gan.train(epochs=20000, batch_size=32, sample_interval=100, save_sample2dir="../samples/Exp0_gaussian_noise_100_no_batchnorm/exp0", save_weights_path='../Weights/Exp0_gaussian_noise_100_no_batchnorm/exp0.h5', save_model=False)
	# gan.deploy_transform(stop_after=200)
	# gan.deploy_transform(stop_after=400, save2file="../domain_adapted/Exp7/generated.npy")
	# gan.deploy_debug(save2file="../domain_adapted/WGAN_GP/Exp4/debug_sobol.npy", sample_size=100, noise_number=256, seed = 17)
	
	# gan.deploy_transform(save2file="../domain_adapted/MNIST_M/{}/generated.npy".format(EXP_NAME), noise_range=5.0, separate_cls=False, stop_after=None, seed=17)

	# gan.deploy_classification()


	# gan.deploy_cherry_pick(linspace_size=2.5)
	# gan.deploy_demo_only()
	#########################
	#      Good example
	#########################
	# gan.load_pretrained_weights(weights_path='../Weights/WGAN_GP/Exp4_11/Exp0.h5')
	# gan.deploy_debug(save2file="../domain_adapted/WGAN_GP/Exp4_14_1/debug_linear.npy", 
	# 	sample_size=100, 
	# 	noise_number=512, 
	# 	use_sobol=False, 
	# 	use_linear=True, 
	# 	use_sphere=False,
	# 	use_uniform_linear=False,
	# 	use_zeros=False,
	# 	seed = 17)