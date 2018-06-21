


import numpy as np 
import matplotlib.pyplot as plt
import os, sys
sys.path.append("/home/lulin/na4/my_packages")
sys.path.append("/home/lulin/Desktop/Desktop/Python_projets/my_packages")
from utils import generator


def plot_D_statistic(history, show=False, cut=None, save2dir="../results/"):
	if not os.path.exists(save2dir):
		os.makedirs(save2dir)
	if cut is not None:
		history = history[:cut]
	length = len(history)
	xaxis_scale = np.arange(0, length*10, 10)
	plt.figure()
	plt.title("Critic loss")
	plt.plot(xaxis_scale, history[:, 0], label="WGAN-GP loss")
	plt.plot(xaxis_scale, history[:, 1], label="real imgs loss")
	plt.plot(xaxis_scale, history[:, 2], label="fake imgs loss")
	plt.plot(xaxis_scale, history[:, 3], label="GP penalization loss")
	plt.xlabel("Iteration")
	plt.legend(loc="best")
	plt.savefig(os.path.join(save2dir, "critic_loss.png"))
	if show:
		plt.show()
	plt.close()

	plt.figure()
	plt.title("Critic accuracy")
	plt.plot(xaxis_scale, history[:, 4], label="Critic acc (real)")
	plt.plot(xaxis_scale, history[:, 5], label="Critic acc (fake)")
	plt.xlabel("Iteration")
	plt.legend(loc="best")
	plt.savefig(os.path.join(save2dir, "critic_acc.png"))
	if show:
		plt.show()
	plt.close()
def plot_D_statistic_SN(history, show=False, cut=None, save2dir="../results/"):
	if not os.path.exists(save2dir):
		os.makedirs(save2dir)
	if cut is not None:
		history = history[:cut]
	length = len(history)
	xaxis_scale = np.arange(0, length*10, 10)
	plt.figure()
	plt.title("Critic loss")
	plt.plot(xaxis_scale, history[:, 0], label="WGAN-GP loss")
	plt.plot(xaxis_scale, history[:, 1], label="real imgs loss")
	plt.plot(xaxis_scale, history[:, 2], label="fake imgs loss")
	plt.xlabel("Iteration")
	plt.legend(loc="best")
	plt.savefig(os.path.join(save2dir, "critic_loss.png"))
	if show:
		plt.show()
	plt.close()

	plt.figure()
	plt.title("Critic accuracy")
	plt.plot(xaxis_scale, history[:, 3], label="Critic acc (real)")
	plt.plot(xaxis_scale, history[:, 4], label="Critic acc (fake)")
	plt.xlabel("Iteration")
	plt.legend(loc="best")
	plt.savefig(os.path.join(save2dir, "critic_acc.png"))
	if show:
		plt.show()
	plt.close()
def plot_G_statistic(history, show=False, save2dir="../results/"):
	if not os.path.exists(save2dir):
		os.makedirs(save2dir)
	length = len(history)
	last_100_mean_cls_acc = np.mean(history[-100:, 4])

	plt.figure()
	plt.title("Generator loss")
	plt.plot(history[:, 0], label="WGAN loss")
	plt.xlabel("Iteration")
	plt.legend(loc="best")
	plt.savefig(os.path.join(save2dir, "generator_loss.png"))
	if show:
		plt.show()
	plt.close()

	plt.figure()
	plt.title("Classifier accuracy")
	plt.plot(history[:, 4], label="Cls acc")
	plt.plot(last_100_mean_cls_acc*np.ones(length), label="Last mean acc: {}".format(last_100_mean_cls_acc))
	plt.xlabel("Iteration")
	plt.legend(loc="best")
	plt.savefig(os.path.join(save2dir, "classifier_acc.png"))
	if show:
		plt.show()
	plt.close()



def plot_G_statistic_seg(history, show=False, cut=None, save2dir="../results/"):
	if not os.path.exists(save2dir):
		os.makedirs(save2dir)
	if cut is not None:
		history = history[:cut]
	length = len(history)
	# last_100_mean_seg_acc = np.mean(history[-100:, 4])
	xaxis_scale = np.arange(0, length*10, 10)
	plt.figure()
	plt.title("Generator loss")
	plt.plot(xaxis_scale, history[:, 0], label="WGAN loss")
	plt.xlabel("Iteration")
	plt.legend(loc="best")
	plt.savefig(os.path.join(save2dir, "generator_loss.png"))
	if show:
		plt.show()
	plt.close()

	plt.figure()
	plt.title("Segmenter accuracy")
	# plt.set_xticks(np.arange(0, length*10, 10))
	best_dice = np.max(history[:, -1])
	plt.plot(xaxis_scale, history[:, 4], label="Seg acc (train)")
	plt.plot(xaxis_scale, 1.-history[:, 2], label="Seg dice (train)")
	plt.plot(xaxis_scale, history[:, -2]/100, label="Seg dice (current)")
	plt.plot(xaxis_scale, (best_dice/100)*np.ones(length) , label="Best mean dice {:.2f}%".format(best_dice))
	plt.plot(xaxis_scale, history[:, -1]/100, label="Seg dice (test)")
	# plt.plot(last_100_mean_cls_acc*np.ones(length), label="Last mean acc: {}".format(last_100_mean_cls_acc))

	plt.xlabel("Iteration")
	plt.legend(loc="best")
	plt.savefig(os.path.join(save2dir, "segmenter_acc.png"))
	if show:
		plt.show()
	plt.close()

def plot_intensity_stat(history, show=False, save2dir="../results/"):
	
	length = len(history)
	xaxis_scale = np.arange(0, length*10, 10) # 10, TODO
	plt.figure()
	plt.title("Liver intensity statistic")
	plt.plot(xaxis_scale, history[:, 0], label="Mean intensity")
	plt.plot(xaxis_scale, history[:, 1], label="Std intensity")
	plt.xlabel("Iteration")
	plt.legend(loc="best")
	plt.savefig(os.path.join(save2dir, "intensity.png"))
	plt.close()
def plot_spectra_stat(history, show=False, save2dir="../results/"):
	length = len(history)
	num_layers = history.shape[1]
	xaxis_scale = np.arange(0, length*10, 10) # 10, TODO
	plt.figure()
	plt.title("Singular value of layers")
	for i in range(num_layers):
		plt.plot(xaxis_scale, history[:, i], label="Singular value of layer {}".format(i))
	plt.xlabel("Iteration")
	plt.legend(loc="best")
	plt.savefig(os.path.join(save2dir, "spectra.png"))
	plt.close()








def download_image_for_gif(imgs_path="../domain_adapted/WGAN_GP/Exp4/debug_uniform_linear.npy"):
	from tqdm import tqdm 
	exp_name = imgs_path.split("/")[-2]
	domain_adapted_images = np.load(imgs_path)
	domain_adapted_images = (domain_adapted_images+1)/2.
	
	r = 5
	c = 15
	num_samples = domain_adapted_images.shape[0]
	num_noises = domain_adapted_images.shape[1]
	for batch_imgs in tqdm(range(int(num_samples/5))):
		for batch_noise in range(int(num_noises/15)-1):
			dirpath = os.path.join("../domain_adapted/collections", exp_name, str(batch_imgs))
			if not os.path.exists(dirpath):
				os.makedirs(dirpath)

			fig, axs = plt.subplots(r, c, figsize=(5*c, 5*r))
			for j in range(r):
				for i in range(c):
					axs[j,i].imshow(domain_adapted_images[j+batch_imgs*r][i+batch_noise*c])
					axs[j,i].axis('off')
			plt.savefig(os.path.join(dirpath, "{}.png".format(batch_noise)))
			plt.close()

def download_image_for_gif2(imgs_path="../domain_adapted/WGAN_GP/Exp4/demo.npy"):
	from tqdm import tqdm 
	from time import time
	print("Loading file...")
	st = time()
	exp_name = imgs_path.split("/")[-2]
	domain_adapted_images = np.load(imgs_path)
	domain_adapted_images = (domain_adapted_images+1)/2.
	print("+ Done.")
	print("Elapsed time {}".format(time()-st))
	r = 5
	c = 5
	print(domain_adapted_images.shape)
	num_noises = domain_adapted_images.shape[0]
	for batch_noise in tqdm(range(num_noises)):	
		dirpath = "../domain_adapted/collections/Exp4_13"
		if not os.path.exists(dirpath):
			os.makedirs(dirpath)

		fig, axs = plt.subplots(r, c, figsize=(5*c, 5*r))
		for j in range(r):
			for i in range(c):
				axs[j,i].imshow(domain_adapted_images[batch_noise][c*j+i])
				axs[j,i].axis('off')
		plt.savefig(os.path.join(dirpath, "{}.png".format(batch_noise)))
		plt.close()

def make_gif(dirpath="../samples/exp0", save2path="../demo/test_imageio.gif"):
	import imageio
	import cv2
	dirname = "/".join(save2path.split("/")[:-1])
	if not os.path.exists(dirname):
		os.makedirs(dirname)

	collections = []
	for file in generator(root_dir=dirpath, file_type='png', file_label_fun=None, stop_after = None, verbose=1):
		collections.append(file)
		# print(file)
	collections.sort()
	collections = sorted(collections, key=lambda x:int(x.split("/")[-1].split(".")[0]))
	# print(collections)
	print("Reading images...")
	collections = [cv2.imread(x) for x in collections[100:-100]]
	# collections = list(map(lambda file:cv2.imread(file), collections))
	print("+ Done.")
	print("Making Gif...")
	imageio.mimsave(save2path, collections)
	print("+ Done.")
	# import ipdb; ipdb.set_trace()


if __name__=="__main__":
	print("Start")

	# FOLDER_NAME = "Exp4_13"
	# filepath = "../Weights/WGAN_GP/{}/D_Losses.csv".format(FOLDER_NAME)
	# with open(filepath, "rb") as file:
	# 	D = np.loadtxt(file, delimiter=",")
	# print(D.shape)
	# plot_D_statistic(D, save2dir="../results/WGAN_GP/{}".format(FOLDER_NAME))


	# filepath = "../Weights/WGAN_GP/{}/G_Losses.csv".format(FOLDER_NAME)
	# with open(filepath, "rb") as file:
	# 	G = np.loadtxt(file, delimiter=",")
	# print(G.shape)
	# plot_G_statistic(G, save2dir="../results/WGAN_GP/{}".format(FOLDER_NAME))


	
	# import ipdb; ipdb.set_trace()
	# download_image_for_gif()

	# download_image_for_gif2()
	# make_gif(dirpath="../domain_adapted/collections/Exp4_13")
	# print("+ Done.")
	# make_gif(dirpath="../domain_adapted/collections/Exp4/0")



	#################
	## Segmentation 
	#################

	# FOLDER_NAME = "Exp7"
	# filepath = "../Weights/CT2XperCT/{}/G_Losses.csv".format(FOLDER_NAME)
	# with open(filepath, "rb") as file:
	# 	G = np.loadtxt(file, delimiter=",")
	# print(G.shape)
	# plot_G_statistic_seg(G, show=False, save2dir="../results/CT2XperCT/{}".format(FOLDER_NAME))



	# filepath = "../Weights/CT2XperCT/{}/D_Losses.csv".format(FOLDER_NAME)
	# with open(filepath, "rb") as file:
	# 	D = np.loadtxt(file, delimiter=",")
	# print(D.shape)
	# plot_D_statistic(D, show=False, save2dir="../results/CT2XperCT/{}".format(FOLDER_NAME))
	
