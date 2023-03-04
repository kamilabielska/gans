# GANs
Generative Adversarial Networks are a really interesting subject to study and it gets even more interesting (and at times quite frustrating) when you get to the implementation part and start tweaking all the hyperparameters in hope that *this* one is going to improve the quality of the generated images. It's quite easy to become a little obsessed with searching for and trying new tips and tricks to improve your GAN model.

I found the most success working with DCGANs (Deep Convolutional GANs), which use binary crossentropy as a loss function, whereas Wasserstein GANs seemed more problematic and overall harder to train. Implementing ProGAN (Progressively Growing GAN) model was quite a journey and, even though the code for sure isn't the nicest, I'm just glad that in the end I got it to work.

This repository provides a framework for working with GANs (`gans` folder): model classes into which one can plug their own generator and discriminator (`models.py`), custom layers (`layers.py`) and callbacks (`callbacks.py`). Usage is demonstrated in `anime_gan.ipynb` notebook.

I focused on generating anime faces using [anime faces dataset](https://www.kaggle.com/datasets/splcher/animefacedataset):
![sample images](https://raw.githubusercontent.com/kamilabielska/gans/main/img/sample_images.jpg)

Here is illustrated the progress of DCGAN on 4 fixed latent vectors (generated images were saved after each of 80 epochs):
![dcgan progress](https://github.com/kamilabielska/gans/blob/main/img/gan_progress.gif?raw=true)

***
**literature**:
- [Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661.pdf), Goodfellow et al., 2014
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf), Radford et al., 2015
- [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/pdf/1701.00160.pdf), Goodfellow, 2017
- [Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf), Gulrajani et al., 2017
- [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/pdf/1710.10196.pdf), Karras et al., 2018
