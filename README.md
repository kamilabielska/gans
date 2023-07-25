# GANs
Generative Adversarial Networks are a really interesting subject to study and it gets even more interesting (and at times quite frustrating) when you get to the implementation part and start tweaking all the hyperparameters in hope that *this* one is going to improve the quality of generated images. It's quite easy to become a little obsessed with searching for and trying new tips and tricks to improve your GAN model.

## DCGAN nad ProGAN
I found the most success working with DCGANs (Deep Convolutional GANs), which use binary crossentropy as a loss function, whereas Wasserstein GANs seemed more problematic and overall harder to train. Implementing ProGAN (Progressively Growing GAN) model was quite a journey and, even though the code for sure isn't the nicest, I'm just glad that in the end I got it to work. Disclaimer: models are not implemented *exactly* as described in the articles, my main goal was to mimic a general concept (e.g. progressive growing). Additional components, particular methods, settings or other details were kept or omitted according to my liking and/or the results they produced.

This repository provides a framework for working with GANs (`gans` folder): model classes into which one can plug their own generator and discriminator (`models.py`), custom layers (`layers.py`) and callbacks (`callbacks.py`). Usage is demonstrated in `anime_gan.ipynb` notebook.

I focused on generating anime faces using [anime faces dataset](https://www.kaggle.com/datasets/splcher/animefacedataset):
![sample images](https://raw.githubusercontent.com/kamilabielska/gans/main/img/sample_images.jpg)

Here is illustrated the progress of DCGAN on 4 fixed latent vectors (generated images were saved after each of 80 epochs):
![dcgan progress](https://github.com/kamilabielska/gans/blob/main/img/gan_progress.gif?raw=true)

And here that of ProGAN (also 4 fixed vectors over 80 epochs):
![progan progress](https://github.com/kamilabielska/gans/blob/main/img/progan_progress.gif?raw=true)

## StyleGAN
I experimented with custom implementations of StyleGAN and StyleGAN2, building on top of the above framework, but the models kept diverging after several (10-20) epochs of training. Additionally, each epoch was taking a long time due to the complexity of the model, so playing with hyperparameters and architecture adjustments became quite wearisome.

Because of that I moved to StyleGAN3 and this time used their [official implementation](https://github.com/NVlabs/stylegan3) available on Github. After some hurdles I managed to set up the environment on Colab and finally got the code to work. However, even here training from scratch and tweaking the hyperparameters did not lead to satisfying results in reasonable time and after using reasonable amount of resources. Transfer learning turned out to be the best option, even though it required upscaling images in my dataset to 256x256 resolution. I used model trained on the FFHQ-U dataset, as it also consists of faces, although real human ones. The Colab set up and training code are available in the `anime_stylegan3.ipynb` notebook (careful, github preview messes up bash cells).

Here is illustrated the process of fine-tuning the model:
![stylegan3 progress](https://github.com/kamilabielska/gans/blob/main/img/stylegan3_progress.gif?raw=true)

And here is a fun interpolation video (how cool is that??):
![stylegan3 interpolation](https://github.com/kamilabielska/gans/blob/main/img/stylegan3_inter.gif?raw=true)

***
**literature**:
- [Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661.pdf), Goodfellow et al., 2014
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf), Radford et al., 2015
- [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/pdf/1701.00160.pdf), Goodfellow, 2017
- [Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf), Gulrajani et al., 2017
- [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/pdf/1710.10196.pdf), Karras et al., 2018
- [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/pdf/1812.04948.pdf), Kerras et al., 2019
- [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/pdf/1912.04958.pdf), Kerras et al., 2020
- [Alias-Free Generative Adversarial Networks](https://arxiv.org/pdf/2106.12423.pdf), Kerras et al., 2021
