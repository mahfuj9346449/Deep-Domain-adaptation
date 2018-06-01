# Deep-Domain-adaptation

Implementation of PixelDA from paper *[Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks](https://arxiv.org/abs/1612.05424)*. This implementation is based on the original implemenation (keras) of [github eriklindernoren/Keras-GAN](https://github.com/eriklindernoren/Keras-GAN#pixelda). However, his code doesn't fit the original paper (both the model design and the accuracy). 


## Results

Results of unsupervised domain adaptation using PixelDA (not yet optimal). 

|           Method          | Accuracy on MNIST-M |
|:-------------------------:|:-------------------:|
|    Source only (MNIST)    |        62.31%       |
| PixelDA (eriklindernoren) |         95%         |
|       PixelDA (with JS)   |      **97.07%**     |
|   PixelDA (with WGAN-GP)  |      **98.60%**     |


In paper, they claim accuracy 98.2% for the classification on MNIST-M (domain adaptation **from MNIST to MNIST-M**). 

With WGAN-GP, I get mean acc **98.60%** (four runs) with 99% confidence interval: [98.38, 98.82].

Project ongoing...


