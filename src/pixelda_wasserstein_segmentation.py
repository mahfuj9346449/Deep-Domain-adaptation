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
		KTF.set_session(get_session(gpu_fraction=0.2)) # NEW 28-9-2017
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
from keras.initializers import he_normal, he_uniform
try:
	from HPOlib_lu.Quasi_Monte_Carlo.sobol_lib import i4_sobol_generate
except:
	print("Can't import Sobol library.")
	pass

from tqdm import tqdm
import dill
from unet.U_net import UNet
from unet.CT_generator import MyDataset
from DLalgors import _DLalgo
import cv2
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

def somme(A):
	return K.sum(K.sum(K.sum(A, axis=-1), axis=-1), axis=-1)


def dice_coef(y_true, y_pred, smooth=0.0001):
	"""
	Dice = (2*|X & Y|)/ (|X|+ |Y|)
		 =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
	ref: https://arxiv.org/pdf/1606.04797v1.pdf
	"""
	intersection = somme((y_true * y_pred))
	return (2. * intersection + smooth) / (somme(y_true) + somme(y_pred) + smooth)

def dice_coef_loss(y_true, y_pred):
	return 1-dice_coef(y_true, y_pred)

def IoU_metric(y_true, y_pred, smooth=0.0001):
	"""
	IoU (intersection over union)

	"""
	intersection = somme((y_true * y_pred)) #K.sum(K.abs(y_true * y_pred), axis=-1)
	sum_ = somme(y_true) + somme(y_pred) #K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
	
	return (intersection + smooth) / (sum_ - intersection + smooth)

def dice_predict(y_gt, y_pred):

	y_gt = np.squeeze(y_gt)
	y_pred = np.squeeze(y_pred)
	
	y_pred = np.round(y_pred)
	intersection = np.sum(np.sum(2*y_pred*y_gt, axis=-1), axis=-1)
	sum_area = np.sum(np.sum(y_pred, axis=-1), axis=-1)+np.sum(np.sum(y_gt, axis=-1), axis=-1)

	dice_all = intersection/sum_area
	return dice_all, np.mean(dice_all)


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
		self.dataset_name = "CT" #"MNIST" # "CT"

		if self.dataset_name == "MNIST":
			self.img_shape = (32, 32, 3)
			self.num_classes = 10
		elif self.dataset_name == "CT":
			self.img_shape = (128, 128, 1)
		else:
			raise ValueError("Only support two datasets for now. ('CT', 'MNIST')")

		self.img_rows, self.img_cols, self.channels = self.img_shape

		self.noise_size = noise_size #(100,)
		self.batch_size = batch_size
		# Loss weights
		self.lambda_adv = 5#before Exp5:10 # Exp1: 20 #17 MNIST-M
		self.lambda_seg = 1
		# Number of filters in first layer of discriminator and Segmenter
		self.df = 64 
		self.sf = 64

		self.normalize_G = False
		self.normalize_D = False
		self.normalize_S = False
		
		# Number of residual blocks in the generator
		self.residual_blocks = 12 #before Exp5: 6 #17 # 6 # NEW TODO 14/5/2018
		self.use_PatchGAN = use_PatchGAN #False
		self.use_Wasserstein = use_Wasserstein
		self.use_He_initialization = False
		self.my_initializer = lambda :he_normal() if self.use_He_initialization else "glorot_uniform" # TODO

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

	
	def build_all_model(self):

		# optimizer = Adam(0.0002, 0.5)
		optimizer = Adam(0.0001, beta_1=0.5, beta_2=0.9) # Exp4 
		# optimizer = Adam(0.0001, beta_1=0.0, beta_2=0.9) # Exp3 of CT2XperCT

		

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
			self.discriminator_model = Model(inputs=[img_B, fake_img],  #, avg_img
											# loss_weights=[1,1,1], # useless, since we have multiply the penalization by GRADIENT_PENALTY_WEIGHT=10
											outputs=[real_img_rating, fake_img_rating, avg_img_output])
			self.discriminator_model.compile(loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],
				optimizer=optimizer,
				metrics=[my_critic_acc])
		else:
			self.discriminator.compile(loss='mse',
				optimizer=optimizer,
				metrics=['accuracy'])



		# For the combined model we will only train the generator and Segmenter
		self.discriminator.trainable = False

		# Build the generator
		self.generator = self.build_generator()
		self.generator.name = "Generator"
		# Build the task (segmentation) network
		self.seg = self.build_segmenter()
		self.seg.name = "Segmenter" 
		# Input images from both domains

		
		# Input noise
		noise = Input(shape=self.noise_size, name='noise_input')

		# Translate images from domain A to domain B
		fake_B = self.generator([img_A, noise])

		# Segment the translated image
		mask_pred = self.seg(fake_B)

		
		# Discriminator determines validity of translated images
		valid = self.discriminator(fake_B) # fake_B_rating
		if self.use_Wasserstein:
			self.combined = Model(inputs=[img_A, noise], outputs=[valid, mask_pred])
			self.combined.compile(optimizer=optimizer, 
									loss=[wasserstein_loss, dice_coef_loss],
									loss_weights=[self.lambda_adv, self.lambda_seg], 
									metrics=['accuracy'])
		else:
			self.combined = Model([img_A, noise], [valid, mask_pred])
			self.combined.compile(loss=['mse', dice_coef_loss],
										loss_weights=[self.lambda_adv, self.lambda_seg],
										optimizer=optimizer,
										metrics=['accuracy'])



	def load_dataset(self, dataset_name="CT", domain_A_folder="output8", domain_B_folder="output5_x_128"):
		self.dataset_name = dataset_name

		if self.dataset_name == "MNIST":
			# Configure MNIST and MNIST-M data loader
			self.data_loader = DataLoader(img_res=(self.img_rows, self.img_cols))
		elif self.dataset_name == "CT":
			bodys_filepath_A = "/home/lulin/na4/src/output/{}/train/bodys.npy".format(domain_A_folder)
			masks_filepath_A = "/home/lulin/na4/src/output/{}/train/liver_masks.npy".format(domain_A_folder)
			self.Dataset_A = MyDataset(paths=[bodys_filepath_A, masks_filepath_A], batch_size=self.batch_size, augment=False, seed=17, domain="A")

			bodys_filepath_B = "/home/lulin/na4/src/output/{}/train/bodys.npy".format(domain_B_folder)
			masks_filepath_B = "/home/lulin/na4/src/output/{}/train/liver_masks.npy".format(domain_B_folder)
			self.Dataset_B = MyDataset(paths=[bodys_filepath_B, masks_filepath_B], batch_size=self.batch_size, augment=False, seed=17, domain="B")
		else:
			pass
		

	def build_generator(self):
		"""Resnet Generator"""

		def residual_block(layer_input, normalization=self.normalize_G):
			"""Residual block described in paper"""
			d = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer=self.my_initializer())(layer_input)
			if normalization:
				d = InstanceNormalization()(d)
				# d = BatchNormalization(momentum=0.8)(d) #  6/5/2018
			d = Activation('relu')(d)
			d = Conv2D(64, kernel_size=3, strides=1, padding='same')(d)
			if normalization:
				d = InstanceNormalization()(d)
				# d = BatchNormalization(momentum=0.8)(d) #  6/5/2018
			d = Add()([d, layer_input])
			return d

		# Image input
		img = Input(shape=self.img_shape, name='image_input')

		## Noise input
		noise = Input(shape=self.noise_size, name='noise_input')
		noise_layer = Dense(self.img_rows*self.img_cols, activation="relu", kernel_initializer=self.my_initializer())(noise)
		noise_layer = Reshape((self.img_rows,self.img_cols, 1))(noise_layer)
		conditioned_img = keras.layers.concatenate([img, noise_layer])
		# keras.layers.concatenate

		# l1 = Conv2D(64, kernel_size=3, padding='same', activation='relu')(img)
		l1 = Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer=self.my_initializer())(conditioned_img)
		

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
				validity = Dense(1, activation=None)(Flatten()(d4)) # he_normal ?? 
			else:
				validity = Dense(1, activation='sigmoid')(Flatten()(d4))
			

		return Model(img, validity)

	def build_segmenter(self):
		"""Segmenter layer"""
		model = UNet(self.img_shape, depth=3, dropout=0.5, start_ch=32, upconv=False, batchnorm=self.normalize_S)
		
		return model

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
		print("Segmenter summary:")
		self.seg.summary()
		
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
		
	def train(self, iterations, 
		sample_interval=50, 
		save_sample2dir="../samples/exp0", 
		save_weights_path='../Weights/all_weights.h5', 
		save_model=False,
		time_monitor=True):
		dirpath = "/".join(save_weights_path.split("/")[:-1])
		if not os.path.exists(dirpath):
			os.makedirs(dirpath)
		self.save_config(save2path=os.path.join(dirpath, "config.dill"), verbose=True)

		# segmentation accuracy on 100 last batches of domain B
		test_accs = []


		## Monitor to save model weights Lu
		best_test_cls_acc = 0
		second_best_cls_acc = -1

		st = time()
		elapsed_time = 0
		for iteration in range(iterations):
			if time_monitor and (iteration%10 ==0) and (iteration>0):
				et = time()
				elapsed_time = et-st
				st = et
				
			# ---------------------
			#  Train Discriminator
			# ---------------------
			# n_sample = half_batch # imgs_A.shape[0]
			
			for _ in range(self.critic_steps):

				if self.dataset_name == "MNIST":
					imgs_A, _ = self.data_loader.load_data(domain="A", batch_size=self.batch_size)
					imgs_B, _ = self.data_loader.load_data(domain="B", batch_size=self.batch_size)	
				elif self.dataset_name == "CT":
					imgs_A, _ = self.Dataset_A.next()
					imgs_B, _ = self.Dataset_B.next()
				
				noise_prior = np.random.normal(0,1, (self.batch_size, self.noise_size[0])) 
				# noise_prior = np.random.rand(half_batch, self.noise_size[0]) #  6/5/2018
				
				# Translate images from domain A to domain B
				fake_B = self.generator.predict([imgs_A, noise_prior])
				if self.use_PatchGAN:
					valid = np.ones((self.batch_size,) + self.disc_patch)
					fake = np.zeros((self.batch_size,) + self.disc_patch)
				else:
					if self.use_Wasserstein:
						valid = np.ones((self.batch_size, 1))
						fake = - valid #np.ones((half_batch, 1)) # = - valid ? 
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
			#  Train Generator and Segmenter
			# --------------------------------
			# Sample a batch of images from both domains

			if self.dataset_name == "MNIST":
				imgs_A, _, masks_A = self.data_loader.load_data(domain="A", batch_size=self.batch_size, return_mask=True)
				imgs_B, _, masks_B = self.data_loader.load_data(domain="B", batch_size=self.batch_size, return_mask=True)
					
			elif self.dataset_name == "CT":
				imgs_A, masks_A = self.Dataset_A.next()
				imgs_B, masks_B = self.Dataset_B.next()
			else:
				pass


			# One-hot encoding of labels
			# labels_A = to_categorical(labels_A, num_classes=self.num_classes)


			# The generators want the discriminators to label the translated images as real
			if self.use_PatchGAN:
				valid = np.ones((self.batch_size,) + self.disc_patch)
			else:
				valid = np.ones((self.batch_size, 1))

			#
			noise_prior = np.random.normal(0,1, (self.batch_size, self.noise_size[0])) 
			# noise_prior = np.random.rand(batch_size, self.noise_size[0]) #  6/5/2018

			# Train the generator and Segmenter
			g_loss = self.combined.train_on_batch([imgs_A, noise_prior], [valid, masks_A]) 


			#-----------------------
			# Evaluation (domain B)
			#-----------------------

			pred_B = self.seg.predict(imgs_B)
			# test_acc = np.mean(np.argmax(pred_B, axis=1) == labels_B)
			
			_, test_acc = dice_predict(masks_B, pred_B) 
	
			# Add accuracy to list of last 100 accuracy measurements
			test_accs.append(test_acc)
			if len(test_accs) > 100:
				test_accs.pop(0)

			
			if iteration % 10 == 0:
				

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

				message = "{} : [D - loss: {:.5f}, GP_loss: {:.5f}, (+) acc: {:.2f}%, (-) acc: {:.2f}%, acc: {:.2f}%], [G - loss: {:.5f}], [seg - loss: {:.5f}, acc: {:.2f}%, test_dice: {:.2f}% ({:.2f}%)]".format(iteration, d_loss[0], d_loss[3], d_real_acc, d_fake_acc, d_train_acc, gen_loss, clf_train_loss, clf_train_acc, current_test_acc, test_mean_acc)

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
				if time_monitor:
					message += "... {:.2f}s.".format(elapsed_time)
				print(message)


			# If at save interval => save generated image samples
			if iteration % sample_interval == 0:
				self.sample_images(iteration, save2dir=save_sample2dir)


		#### NEW 24/5/2018
		self.combined.save_weights(save_weights_path[:-3]+"_final.h5")
			
				

	def sample_images(self, iterations, save2dir="../samples"):
		if not os.path.exists(save2dir):
			os.makedirs(save2dir)

		
		if self.dataset_name == "MNIST":
			r, c = 5, 10
			imgs_A, _ = self.data_loader.load_data(domain="A", batch_size=c)
		elif self.dataset_name == "CT":
			r, c = 2, 5
			assert r == 2
			imgs_A, masks_A = self.Dataset_A.next()
			imgs_A = imgs_A[:c]
			masks_A = masks_A[:c]
			masks_A = np.squeeze(masks_A)
			# raise ValueError("Not implemented error.")
		else:
			pass
		

		n_sample = imgs_A.shape[0] # == c

		gen_imgs = imgs_A
		for i in range(r-1):
			noise_prior = np.random.normal(0,1, (n_sample, self.noise_size[0])) # TODO
			# noise_prior = np.random.normal(0,3, (n_sample, self.noise_size[0])) # TODO # 16/5/2018
			# noise_prior = np.random.rand(n_sample, self.noise_size[0]) # TODO 6/5/2018

			# Translate images to the other domain
			fake_B = self.generator.predict([imgs_A, noise_prior])
			# print(fake_B.shape)
			gen_imgs = np.concatenate([gen_imgs, fake_B])

		if self.dataset_name == "MNIST":
			# Rescale images from (-1, 1) to (0, 1)
			gen_imgs = 0.5 * gen_imgs + 0.5
		elif self.dataset_name == "CT":
			gen_imgs = np.squeeze(gen_imgs)
		# print(gen_imgs.shape)
		#titles = ['Original', 'Translated']

		# TODO
		r = 4
		fig, axs = plt.subplots(r, c, figsize=(3*c, 3*r))

		cnt = 0
		for i in range(2): # replace r by 2
			for j in range(c):
				axs[i,j].imshow(gen_imgs[cnt], cmap="gray")
				#axs[i, j].set_title(titles[i])
				axs[i,j].axis('off')
				cnt += 1
		for j in range(c):
			############ TODO  ############
			# visualize image with adaptive histogram
			axs[2,j].imshow(apply_adapt_hist()(gen_imgs[j+c*1]), cmap="gray")
			axs[2,j].axis('off')	
			# mask image with ground truth mask
			axs[3,j].imshow(apply_adapt_hist()(gen_imgs[j+c*1]), cmap="gray")
			axs[3,j].imshow(masks_A[j], aspect="equal", cmap="Blues", alpha=0.4)
			axs[3,j].axis('off')
				
		fig.savefig(os.path.join(save2dir, "{}.png".format(iterations)))
		plt.close()

	def train_segmenter(self, iterations, batch_size=32, noise_range=5, save_weights_path=None):
		raise ValueError("Not modified yet.")
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


	def deploy_transform(self, save2file="../domain_adapted/generated.npy", stop_after=None):
		raise ValueError("Not modified yet.")
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

	def deploy_debug(self, save2file="../domain_adapted/debug.npy", sample_size=100, noise_number=128, 
		use_sobol=False, 
		use_linear=False, 
		use_sphere=False, 
		use_uniform_linear=False, 
		use_zeros=False,
		seed = 17):
		raise ValueError("Not modified yet.")
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
		imgs_A, labels_A = self.data_loader.load_data(domain="A", batch_size=sample_size)

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
	
	def deploy_segmentation(self, batch_size=32):
		print("Predicting ... ")
		if self.dataset_name == "MNIST":
			pred_B = self.seg.predict(self.data_loader.mnistm_X, batch_size=batch_size)
			precision = (np.argmax(pred_B, axis=1) == self.data_loader.mnistm_y)
			Moy = np.mean(precision)
			Std = np.std(precision)
		elif self.dataset_name == "CT":
			pred_B = self.seg.predict(self.Dataset_B.X_train, batch_size=batch_size)
			gt_B = self.Dataset_B.Y_train
			dice_all, dice_mean = dice_predict(gt_B, pred_B)
			Moy = dice_mean
			Std = np.std(dice_all)
		print("+ Done.")
			
		N_samples = len(pred_B)
		

		lower_bound = Moy - 2.576*Std/np.sqrt(N_samples) 
		upper_bound = Moy + 2.576*Std/np.sqrt(N_samples)
		print("="*50)
		print("Unsupervised MNIST-M segmentation accuracy : {}".format(Moy))
		print("Confidence interval (99%) [{}, {}]".format(lower_bound, upper_bound))
		print("Length of confidence interval 99%: {}".format(upper_bound-lower_bound))
		print("="*50)
		print("+ All done.")

	def deploy_demo_only(self, save2file="../domain_adapted/WGAN_GP/Exp4/demo.npy", sample_size=25, noise_number=512, linspace_size=10.0):
		raise ValueError("Not modified yet.")
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
		raise ValueError("Not modified yet.")
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


def apply_adapt_hist(clipLimit=2.0, tileGridSize=(8, 8)):
	clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
	def adapt_hist_transform(img):
		## cv2 accept only dtype=uint8 image ! 

		lower = np.min(img)
		img = img -lower
		upper = np.max(img)
		img /= upper
		img *= 255.
		img = img.astype(np.uint8)

		cl1 = clahe.apply(img) # cl1.dtype = float64 !!! [IM]
		# rescale pixel value to range [0, 1]
		cl1 = (cl1/255.)
		return cl1
	return adapt_hist_transform

if __name__ == '__main__':
	gan = PixelDA(noise_size=(100,), use_PatchGAN=False, use_Wasserstein=True, batch_size=64)#32
	# gan.load_config(verbose=True, from_file="../Weights/MNIST_SEG/Exp1/config.dill")
	# gan.load_config(verbose=True, from_file="../Weights/CT2XperCT/Exp_test/config.dill")
	gan.build_all_model()
	gan.summary()
	gan.load_dataset(dataset_name="CT", domain_A_folder="output16", domain_B_folder="output16_x_128")
	gan.print_config()
	# gan.write_tensorboard_graph()
	##### gan.save_config(verbose=True, save2path="../Weights/WGAN_GP/Exp4_7/config.dill")
	# gan.load_pretrained_weights(weights_path='../Weights/CT2XperCT/Exp2/Exp0_bis.h5')
	
	try:
		save_weights_path = '../Weights/CT2XperCT/Exp6/Exp0.h5'
		gan.train(iterations=100000, sample_interval=50, save_sample2dir="../samples/CT2XperCT/Exp6", save_weights_path=save_weights_path)
	except KeyboardInterrupt:
		gan.combined.save_weights(save_weights_path[:-3]+"_keyboardinterrupt.h5")
		sys.exit()
	except:
		gan.combined_GC.save_weights(save_weights_path[:-3]+"_unkownerror.h5")
		raise
	
	####### MNIST-M segmentation
	# gan.load_pretrained_weights(weights_path='../Weights/WGAN_GP/Exp4_14_1/Exp0.h5')
	# gan.load_pretrained_weights(weights_path='../Weights/MNIST_SEG/Exp1/Exp0.h5')
	# import ipdb; ipdb.set_trace()
	# gan.train(iterations=100000, sample_interval=50, save_sample2dir="../samples/MNIST_SEG/Exp5", save_weights_path='../Weights/MNIST_SEG/Exp5/Exp0.h5')
	# gan.train_segmenter(iterations=100000, batch_size=64, noise_range=1, save_weights_path="../Weights/MNIST_SEG/Exp5_seg/Exp0.h5")
	# gan.combined.save('../Weights/MNIST_SEG/Exp1/Exp0_model.h5')


	####### MNIST-M Classification (semi-supervised)
	# gan.train(iterations=100000, batch_size=64, sample_interval=100, save_sample2dir="../samples/WGAN_GP/Exp4_13", save_weights_path='../Weights/WGAN_GP/Exp4_13/Exp0.h5')
	
	# gan.deploy_transform(stop_after=200)
	# gan.deploy_transform(stop_after=400, save2file="../domain_adapted/Exp7/generated.npy")
	# gan.deploy_debug(save2file="../domain_adapted/WGAN_GP/Exp4/debug_sobol.npy", sample_size=100, noise_number=256, seed = 17)
	
	# gan.deploy_segmentation()


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