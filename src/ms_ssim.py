import numpy as np 
import os

from scipy import signal
from scipy.ndimage.filters import convolve
from collections import defaultdict

def ssim(img1, img2, cs_map=False):
	"""Return the Structural Similarity Map corresponding to input images img1 
	and img2 (images are assumed to be uint8)
	
	This function attempts to mimic precisely the functionality of ssim.m a 
	MATLAB provided by the author's of SSIM
	https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
	"""
	img1 = img1.astype(np.float64)
	img2 = img2.astype(np.float64)
	size = 11
	sigma = 1.5
	window = gauss.fspecial_gauss(size, sigma)
	K1 = 0.01
	K2 = 0.03
	L = 255 #bitdepth of image
	C1 = (K1*L)**2
	C2 = (K2*L)**2
	mu1 = signal.fftconvolve(window, img1, mode='valid')
	mu2 = signal.fftconvolve(window, img2, mode='valid')
	mu1_sq = mu1*mu1
	mu2_sq = mu2*mu2
	mu1_mu2 = mu1*mu2
	sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
	sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
	sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
	if cs_map:
		return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
					(sigma1_sq + sigma2_sq + C2)), 
				(2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
	else:
		return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
					(sigma1_sq + sigma2_sq + C2))
def msssim(img1, img2):
	"""This function implements Multi-Scale Structural Similarity (MSSSIM) Image 
	Quality Assessment according to Z. Wang's "Multi-scale structural similarity 
	for image quality assessment" Invited Paper, IEEE Asilomar Conference on 
	Signals, Systems and Computers, Nov. 2003 
	
	Author's MATLAB implementation:-
	http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
	"""
	level = 5
	weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
	downsample_filter = np.ones((2, 2))/4.0
	im1 = img1.astype(np.float64)
	im2 = img2.astype(np.float64)
	mssim = np.array([])
	mcs = np.array([])
	for l in range(level):
		ssim_map, cs_map = ssim(im1, im2, cs_map=True)
		mssim = np.append(mssim, ssim_map.mean())
		mcs = np.append(mcs, cs_map.mean())
		filtered_im1 = ndimage.filters.convolve(im1, downsample_filter, 
												mode='reflect')
		filtered_im2 = ndimage.filters.convolve(im2, downsample_filter, 
												mode='reflect')
		im1 = filtered_im1[::2, ::2]
		im2 = filtered_im2[::2, ::2]
	return (np.prod(mcs[0:level-1]**weight[0:level-1])*
					(mssim[level-1]**weight[level-1]))


def _FSpecialGauss(size, sigma):
	"""Function to mimic the 'fspecial' gaussian MATLAB function."""
	radius = size // 2
	offset = 0.0
	start, stop = -radius, radius + 1
	if size % 2 == 0:
		offset = 0.5
		stop -= 1
	x, y = np.mgrid[offset + start:stop, offset + start:stop]
	assert len(x) == size
	g = np.exp(-((x**2 + y**2)/(2.0 * sigma**2)))
	return g / g.sum()


def _SSIMForMultiScale(img1, img2, max_val=255, filter_size=11,
					   filter_sigma=1.5, k1=0.01, k2=0.03):
	"""Return the Structural Similarity Map between `img1` and `img2`.
	This function attempts to match the functionality of ssim_index_new.m by
	Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
	Arguments:
		img1: Numpy array holding the first RGB image batch.
		img2: Numpy array holding the second RGB image batch.
		max_val: the dynamic range of the images (i.e., the difference between the
		  maximum the and minimum allowed values).
		filter_size: Size of blur kernel to use (will be reduced for small images).
		filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
		  for small images).
		k1: Constant used to maintain stability in the SSIM calculation (0.01 in
		  the original paper).
		k2: Constant used to maintain stability in the SSIM calculation (0.03 in
		  the original paper).
	Returns:
		Pair containing the mean SSIM and contrast sensitivity between `img1` and
		`img2`.
	Raises:
		RuntimeError: If input images don't have the same shape or don't have four
		  dimensions: [batch_size, height, width, depth].
	"""
	if img1.shape != img2.shape:
		raise RuntimeError('Input images must have the same shape (%s vs. %s).', img1.shape, img2.shape)
	if img1.ndim != 4:
		raise RuntimeError('Input images must have four dimensions, not %d', img1.ndim)

	img1 = img1.astype(np.float64)
	img2 = img2.astype(np.float64)
	_, height, width, _ = img1.shape

	# Filter size can't be larger than height or width of images.
	size = min(filter_size, height, width)

	# Scale down sigma if a smaller filter size is used.
	sigma = size * filter_sigma / filter_size if filter_size else 0

	if filter_size:
		window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
		mu1 = signal.fftconvolve(img1, window, mode='valid')
		mu2 = signal.fftconvolve(img2, window, mode='valid')
		sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
		sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
		sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
	else:
		# Empty blur kernel so no need to convolve.
		mu1, mu2 = img1, img2
		sigma11 = img1 * img1
		sigma22 = img2 * img2
		sigma12 = img1 * img2

	mu11 = mu1 * mu1
	mu22 = mu2 * mu2
	mu12 = mu1 * mu2
	sigma11 -= mu11
	sigma22 -= mu22
	sigma12 -= mu12

	# Calculate intermediate values used by both ssim and cs_map.
	c1 = (k1 * max_val) ** 2
	c2 = (k2 * max_val) ** 2
	v1 = 2.0 * sigma12 + c2
	v2 = sigma11 + sigma22 + c2
	ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
	cs = np.mean(v1 / v2)
	return ssim, cs

def MultiScaleSSIM(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5,
				   k1=0.01, k2=0.03, weights=None):
	"""Return the MS-SSIM score between `img1` and `img2`.
	This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
	Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
	similarity for image quality assessment" (2003).
	Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
	Author's MATLAB implementation:
	http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

	Arguments:
		img1: Numpy array holding the first RGB image batch.
		img2: Numpy array holding the second RGB image batch.
		max_val: the dynamic range of the images (i.e., the difference between the
		  maximum the and minimum allowed values).
		filter_size: Size of blur kernel to use (will be reduced for small images).
		filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
		  for small images).
		k1: Constant used to maintain stability in the SSIM calculation (0.01 in
		  the original paper).
		k2: Constant used to maintain stability in the SSIM calculation (0.03 in
		  the original paper).
		weights: List of weights for each level; if none, use five levels and the
		  weights from the original paper.
	Returns:
		MS-SSIM score between `img1` and `img2`.
	Raises:
		RuntimeError: If input images don't have the same shape or don't have four
		  dimensions: [batch_size, height, width, depth].
	"""
	bizarre = False
	if img1.shape != img2.shape:

		raise RuntimeError('Input images must have the same shape (%s vs. %s).', img1.shape, img2.shape)
	if img1.ndim != 4:
		raise RuntimeError('Input images must have four dimensions, not %d', img1.ndim)

	# Note: default weights don't sum to 1.0 but do match the paper / matlab code.
	weights = np.array(weights if weights else
					 [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
	levels = weights.size
	downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
	im1, im2 = [x.astype(np.float64) for x in [img1, img2]]
	mssim = np.array([])
	mcs = np.array([])
	for _ in range(levels):
		ssim, cs = _SSIMForMultiScale(
			im1, im2, max_val=max_val, filter_size=filter_size,
			filter_sigma=filter_sigma, k1=k1, k2=k2)
		mssim = np.append(mssim, ssim)
		if cs<0:
			bizarre = True
		mcs = np.append(mcs, np.abs(cs)) ## Modified by Lu ## cs --> np.abs(cs)
		filtered = [convolve(im, downsample_filter, mode='reflect')
					for im in [im1, im2]]
		im1, im2 = [x[:, ::2, ::2, :] for x in filtered]
	# print(weights[0:levels-1])
	# print(weights[levels-1])
	# print(mcs[0:levels-1])
	# print(mssim[levels-1])
	return (np.prod(mcs[0:levels-1] ** weights[0:levels-1])*(mssim[levels-1] ** weights[levels-1])), bizarre


def measure_MNIST_m_msssim(repeat=100, batch_size=10000, separate_cls=True):
	from time import time
	from tqdm import tqdm
	RootPath = os.path.expanduser("~")
	print("Loading MNIST-M datasets...")
	MNIST_M = np.load(os.path.join(RootPath, ".keras/datasets", "mnistm_x.npy"))
	if separate_cls:
		MNIST_M_labels = np.load(os.path.join(RootPath, ".keras/datasets", "mnistm_y.npy"))
	print("+ Done.")
	MNIST_M = MNIST_M - np.min(MNIST_M)
	max_val = np.max(MNIST_M)
	print("Maximum pixel-value of datasets: {}".format(max_val))
	print("Minimum pixel-value of datasets: {}".format(np.min(MNIST_M)))
	num = len(MNIST_M)

	### Random select 100 images
	if not separate_cls:
		st = time()
		score = []
		for ind in tqdm(range(repeat)):
			# print("="*50)
			index = np.random.choice(num, 2*batch_size, replace=False)
			imgs1 = MNIST_M[index[:batch_size]]
			imgs2 = MNIST_M[index[batch_size:]]
			ssim_score, flag = MultiScaleSSIM(imgs1, imgs2, max_val=max_val)
			if flag:
				print(ind)
			score.append(ssim_score)

		print("Elapsed time for MS-SSIM evaluation: {}".format(time()-st))
		print(score)
		
		score = np.array(score)
		moy = np.mean(score)
		std = np.std(score)

		print("MS-SSIM simularity of MNIST-M (mean): {}".format(moy))
		print("Confidence interval (99%): [{}, {}]".format(moy-2.576*std/np.sqrt(repeat), moy+2.576*std/np.sqrt(repeat)))
		return score
	else:
		st = time()
		score = defaultdict(lambda:[])
		for ind in tqdm(range(repeat)):
			for mnist_cls in range(10):
				mnistm_x = MNIST_M[MNIST_M_labels==mnist_cls]
				num = len(mnistm_x)
				if 2*batch_size>=num:

					batch_size = int(num/2)
					print("MNISTM class {} has only {} samples. Reduce batch size to {}".format(mnist_cls, num, batch_size))

				index = np.random.choice(num, 2*batch_size, replace=False)
				imgs1 = mnistm_x[index[:batch_size]]
				imgs2 = mnistm_x[index[batch_size:]]
				ssim_score, flag = MultiScaleSSIM(imgs1, imgs2, max_val=max_val)
				# if flag:
				# 	print(ind)
				score[mnist_cls].append(ssim_score)

		print("Elapsed time for MS-SSIM evaluation: {}".format(time()-st))
		# print(score)
		
		for key in score:

			score_per_cls = np.array(score[key])
			moy = np.mean(score_per_cls)
			std = np.std(score_per_cls)
			print("="*50)
			print("Class {} has MS-SSIM mean score {}".format(key, moy))
			print("Confidence interval (99%): [{}, {}]".format(moy-2.576*std/np.sqrt(repeat), moy+2.576*std/np.sqrt(repeat)))

		return score

def measure_DA_MNISTm_msssim_from_file(filepath="",repeat=100, batch_size=10000, separate_cls=True):
	print("Estimate MS-SSIM score for file: {}".format(filepath))
	from time import time
	from tqdm import tqdm
	RootPath = os.path.expanduser("~")
	print("Loading domain adapted MNIST-M datasets...")
	MNIST_M_generated = np.load(os.path.join(RootPath, ".keras/datasets", "mnistm_x.npy"))
	if separate_cls:
		MNIST_M_labels = np.load(os.path.join(RootPath, ".keras/datasets", "mnistm_y.npy"))
	print("+ Done.")
	MNIST_M_generated = MNIST_M_generated - np.min(MNIST_M_generated)
	max_val = np.max(MNIST_M_generated)
	print("Maximum pixel-value of datasets: {}".format(max_val))
	print("Minimum pixel-value of datasets: {}".format(np.min(MNIST_M_generated)))
	num = len(MNIST_M_generated)

	### Random select 100 images
	if not separate_cls:
		st = time()
		score = []
		for ind in tqdm(range(repeat)):
			# print("="*50)
			index = np.random.choice(num, 2*batch_size, replace=False)
			imgs1 = MNIST_M_generated[index[:batch_size]]
			imgs2 = MNIST_M_generated[index[batch_size:]]
			ssim_score, flag = MultiScaleSSIM(imgs1, imgs2, max_val=max_val)
			if flag:
				print(ind)
			score.append(ssim_score)

		print("Elapsed time for MS-SSIM evaluation: {}".format(time()-st))
		print(score)
		
		score = np.array(score)
		moy = np.mean(score)
		std = np.std(score)

		print("MS-SSIM simularity of MNIST-M (mean): {}".format(moy))
		print("Confidence interval (99%): [{}, {}]".format(moy-2.576*std/np.sqrt(repeat), moy+2.576*std/np.sqrt(repeat)))
		return score
	else:
		st = time()
		score = defaultdict(lambda:[])
		for ind in tqdm(range(repeat)):
			for mnist_cls in range(10):
				mnistm_x = MNIST_M_generated[MNIST_M_labels==mnist_cls]
				num = len(mnistm_x)
				if 2*batch_size>=num:

					batch_size = int(num/2)
					print("MNISTM class {} has only {} samples. Reduce batch size to {}".format(mnist_cls, num, batch_size))

				index = np.random.choice(num, 2*batch_size, replace=False)
				imgs1 = mnistm_x[index[:batch_size]]
				imgs2 = mnistm_x[index[batch_size:]]
				ssim_score, flag = MultiScaleSSIM(imgs1, imgs2, max_val=max_val)
				# if flag:
				# 	print(ind)
				score[mnist_cls].append(ssim_score)

		print("Elapsed time for MS-SSIM evaluation: {}".format(time()-st))
		# print(score)
		
		for key in score:

			score_per_cls = np.array(score[key])
			moy = np.mean(score_per_cls)
			std = np.std(score_per_cls)
			print("="*50)
			print("Class {} has MS-SSIM mean score {}".format(key, moy))
			print("Confidence interval (99%): [{}, {}]".format(moy-2.576*std/np.sqrt(repeat), moy+2.576*std/np.sqrt(repeat)))

		return score

if __name__=="__main__":
	print("Start")

	### Usage
	A = np.ones((5,128,128,3))
	B = np.random.random((5,128,128,3))
	C = np.zeros((5,128,128,3))
	print(MultiScaleSSIM(A,B, max_val=1.0))
	print(MultiScaleSSIM(A,C, max_val=1.0))
	print(MultiScaleSSIM(B,C, max_val=1.0))
	print("="*50)
	np.random.shuffle(B) ## MS-SSIM invariant by order of batch, since it takes np.mean
	print(MultiScaleSSIM(A,B, max_val=1.0))
	print(MultiScaleSSIM(B,C, max_val=1.0))


	print("="*50)
	# measure_MNIST_m_msssim(repeat=50, batch_size=1000, separate_cls=True)
	measure_DA_MNISTm_msssim_from_file(filepath="../domain_adapted/MNIST_M/Exp4_11/generated.npy",repeat=10, batch_size=1000, separate_cls=True)





	### Others method:


	### With tensorflow (however "tf.image.ssim_multiscale" doesn't work ! (1/6/2018, version 1.8.0))
	
	"""
	import keras
	import tensorflow as tf 
	import keras.backend as K

	A = cv2.imread("./1.png")[np.newaxis, :,:,:]
	B = cv2.imread("./2.png")[np.newaxis, :,:,:]
	A = tf.Variable(A/255)
	B = tf.Variable(B/255)
	ssim1 = tf.image.ssim(im1, im2, max_val=1.0)
	#### ssim2 = tf.image.ssim_multiscale(im1, im2, max_val=1.0) # Doesn't work !!!
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		a = sess.run([ssim1])
		print(a)

	"""
	

