


import numpy as np 
import matplotlib.pyplot as plt
import os

def plot_D_statistic(history, show=False, save2dir="../results/"):
	if not os.path.exists(save2dir):
		os.makedirs(save2dir)

	plt.figure()
	plt.title("Critic loss")
	plt.plot(history[:, 0], label="WGAN-GP loss")
	plt.plot(history[:, 1], label="real imgs loss")
	plt.plot(history[:, 2], label="fake imgs loss")
	plt.plot(history[:, 3], label="GP penalization loss")
	plt.xlabel("Iteration")
	plt.legend(loc="best")
	plt.savefig(os.path.join(save2dir, "critic_loss.png"))
	if show:
		plt.show()
	plt.close()

	plt.figure()
	plt.title("Critic accuracy")
	plt.plot(history[:, 4], label="Critic acc (real)")
	plt.plot(history[:, 5], label="Critic acc (fake)")
	plt.xlabel("Iteration")
	plt.legend(loc="best")
	plt.savefig(os.path.join(save2dir, "critic_acc.png"))
	if show:
		plt.show()
	plt.close()

def plot_G_statistic(history, show=False, save2dir="../results/"):
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


if __name__=="__main__":
	print("Start")

	FOLDER_NAME = "Exp4_12"
	filepath = "../Weights/WGAN_GP/{}/D_Losses.csv".format(FOLDER_NAME)
	with open(filepath, "rb") as file:
		D = np.loadtxt(file, delimiter=",")
	print(D.shape)
	plot_D_statistic(D, save2dir="../results/WGAN_GP/{}".format(FOLDER_NAME))


	filepath = "../Weights/WGAN_GP/{}/G_Losses.csv".format(FOLDER_NAME)
	with open(filepath, "rb") as file:
		G = np.loadtxt(file, delimiter=",")
	print(G.shape)
	plot_G_statistic(G, save2dir="../results/WGAN_GP/{}".format(FOLDER_NAME))