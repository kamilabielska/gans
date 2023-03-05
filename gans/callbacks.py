import os
import numpy as np
import tensorflow as tf


class SaveGeneratedImages(tf.keras.callbacks.Callback):
	def __init__(self, data=None, real_data=None, n=4, path=''):
		self.data = data
		self.real_data = real_data
		self.n = n if data is None else tf.shape(data)[0].numpy()
		self.path = path
		self.labels = None

	def on_epoch_end(self, epoch, logs={}):
		image_path = os.path.join(self.path, str(epoch + 1))
		if self.model.conditional:
			self.labels = tf.keras.utils.to_categorical(
				np.random.choice(self.model.n_classes, size=self.n),
				num_classes=self.model.n_classes
			)
		if self.real_data is None:
			self.model.generate_new_samples(
				data=self.data,
				n=self.n,
				out_path=image_path,
				labels=self.labels
			)
		else:
			self.model.generate_and_compare_samples(
				real_data=self.real_data,
				data=self.data,
				n=self.n,
				out_path=image_path,
				labels=self.labels
			)


class GaussianSTDAnnealing(tf.keras.callbacks.Callback):
	def __init__(self, alpha):
		self.alpha = alpha

	def on_train_begin(self, logs=None):
		self.gaussian_layers = []
		for model in [self.model.generator, self.model.discriminator]:
			for layer in model.layers:
				if 'gaussian_noise_annealing' in layer.name:
					self.gaussian_layers.append(layer)
	
	def on_train_batch_begin(self, batch, logs=None):
		for layer in self.gaussian_layers:
			if layer.stddev - self.alpha >= 0:
				layer.stddev.assign_sub(self.alpha)