import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as layers

from tqdm import tqdm


class GAN(tf.keras.Model):
	def __init__(self, generator, discriminator, classes=0, gp_weight=0, **kwargs):
		super(GAN, self).__init__(**kwargs)
		self.generator = generator
		self.discriminator = discriminator
		self.classes = classes
		self.n_classes = len(classes)
		self.conditional = self.n_classes != 0
		self.gp_weight = gp_weight

		self.latent_dim = generator.input_shape[-1] - self.n_classes
		self.image_size = discriminator.input_shape[1]

		self.gen_loss_tracker = tf.keras.metrics.Mean(name='gen_loss')
		self.disc_loss_tracker = tf.keras.metrics.Mean(name='disc_loss')

	@property
	def metrics(self):
		return [
			self.gen_loss_tracker,
			self.disc_loss_tracker
		]

	def compile(self, gen_optimizer, disc_optimizer, loss_type='bce', label_smoothing=False):
		super(GAN, self).compile()
		self.gen_optimizer = gen_optimizer
		self.disc_optimizer = disc_optimizer
		self.label_smoothing = label_smoothing
		self.loss_type = loss_type

		if loss_type == 'bce':
			self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
			self.discriminator_loss = self.discriminator_ce_loss
			self.generator_loss = self.generator_ce_loss

		elif loss_type == 'wgan_gp':
			self.discriminator_loss = self.discriminator_wgan_gp_loss
			self.generator_loss = self.generator_wgan_gp_loss

	def gradient_penalty(self, real_images, fake_images, training=True):
		alpha = tf.random.normal([self.batch_size, 1, 1, 1], 0.0, 1.0)
		diff = fake_images - real_images
		interpolated = real_images + alpha * diff

		with tf.GradientTape() as gp_tape:
			gp_tape.watch(interpolated)
			pred = self.discriminator(interpolated, training=training)

		grads = gp_tape.gradient(pred, [interpolated])[0]
		norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
		gp = tf.reduce_mean((norm - 1.0) ** 2)

		return gp

	def call(self, inputs, training):
		data, noise = inputs

		if self.conditional:
			images, labels = data
			label_channels = self._get_labels_as_channels(labels)

			noise = tf.concat([noise, labels], axis=1)
			data = tf.concat([images, label_channels], axis=-1)

		generated_images = self.generator(noise, training=training)
		real_output = self.discriminator(data, training=training)

		if self.conditional:
			generated_images = tf.concat([generated_images, label_channels], axis=-1)
		fake_output = self.discriminator(generated_images, training=training)

		if self.loss_type == 'wgan_gp':
			gp = self.gradient_penalty(data, generated_images, training)
		else:
			gp = 0

		return real_output, fake_output, gp

	def discriminator_ce_loss(self, real_output, fake_output):
		alpha = 0.9 if self.label_smoothing else 1
		real_loss = self.cross_entropy(alpha*tf.ones_like(real_output), real_output)
		fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
		return real_loss + fake_loss

	def generator_ce_loss(self, fake_output):
		return self.cross_entropy(tf.ones_like(fake_output), fake_output)

	def discriminator_wgan_gp_loss(self, real_output, fake_output):
		real_loss = tf.reduce_mean(real_output)
		fake_loss = tf.reduce_mean(fake_output)
		return fake_loss - real_loss

	def generator_wgan_gp_loss(self, fake_output):
		return -tf.reduce_mean(fake_output)

	def train_step(self, data):
		self.batch_size = tf.shape(data[0] if self.conditional else data)[0]
		noise = tf.random.normal([self.batch_size, self.latent_dim])

		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			real_output, fake_output, gp = self((data, noise), training=True)
			gen_loss = self.generator_loss(fake_output)
			disc_loss = self.discriminator_loss(real_output, fake_output) + gp*self.gp_weight

		gen_grad = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
		disc_grad = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

		self.gen_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables))
		self.disc_optimizer.apply_gradients(zip(disc_grad, self.discriminator.trainable_variables))

		self.gen_loss_tracker.update_state(gen_loss)
		self.disc_loss_tracker.update_state(disc_loss)
		return {
			'gen_loss': self.gen_loss_tracker.result(),
			'disc_loss': self.disc_loss_tracker.result()
		}

	def test_step(self, data):
		self.batch_size = tf.shape(data[0] if self.conditional else data)[0]
		noise = tf.random.normal([self.batch_size, self.latent_dim])

		real_output, fake_output, gp = self((data, noise), training=False)
		gen_loss = self.generator_loss(fake_output)
		disc_loss = self.discriminator_loss(real_output, fake_output) + gp*self.gp_weight
		return {
			'gen_loss': gen_loss,
			'disc_loss': disc_loss
		}

	def _get_labels_as_channels(self, labels):
		label_channels = tf.repeat(
				labels[:, :, None, None], repeats=[self.image_size * self.image_size]
			)
		label_channels = tf.reshape(
			label_channels, (self.batch_size, self.image_size, self.image_size, -1)
		)
		return label_channels
	
	def generate_new_samples(self, data=None, n=4, out_path=None, labels=None):
		if data is None:
			data = tf.random.normal([n, self.latent_dim])
		else:
			n = tf.shape(data)[0].numpy()

		if self.conditional:
			data = tf.concat([data, labels], axis=1)
			string_labels = np.array(self.classes)[np.argmax(labels, axis=1)]
		new_samples = (self.generator.predict(data, verbose=0) + 1) / 2

		fig, axes = plt.subplots(1, n, figsize=(16,5))
		axes = axes.flatten()
		for i in range(n):
			sample = new_samples[i, :].squeeze()
			cmap = 'gray' if np.ndim(sample) == 2 else None
			axes[i].imshow(sample, cmap=cmap)
			axes[i].axis('off')
			if self.conditional:
				axes[i].set_title(string_labels[i])

		if out_path is not None:
			plt.savefig(out_path, bbox_inches='tight')
			plt.close(fig)


class ProGAN(GAN):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def compile(self, building_block, block_kwargs, gen_output_layer,
				disc_input_layer, init_n_filters, drift=0, **kwargs):
		super().compile(**kwargs)
		self.building_block = building_block
		self.block_kwargs = block_kwargs
		self.gen_output_layer = gen_output_layer
		self.disc_input_layer = disc_input_layer
		self.n_filters = init_n_filters
		self.drift = drift

		self.weighted_add_layers = []
		self.factor = 2

	def update_generator(self):
		input_layer = layers.Input(shape=[self.latent_dim + self.n_classes])

		old_generator = None
		for layer in self.generator.layers[1:]:
			if 'to_skip' not in layer.name and layer.name != 'to_rgb':
				if old_generator is None:
					old_generator = layer(input_layer)
				else:
					old_generator = layer(old_generator)
			elif layer.name == 'to_rgb':
				old_to_rgb_output = layer(old_generator)
				layer._name = 'to_skip_old_to_rgb'

		upsampled_old_to_rgb = layers.UpSampling2D(name='to_skip_upsample')(old_to_rgb_output)
		upsampled_old_gen = layers.UpSampling2D()(old_generator)
		
		new_block_output = self.building_block(
			[2*self.n_filters//self.factor, self.n_filters//self.factor],
			pixel_norm=True, **self.block_kwargs)(upsampled_old_gen)
		new_block_output = self.gen_output_layer(name='to_rgb')(new_block_output)

		gen_weighted_add = WeightedAdd(alpha=0.0, name='to_skip_add')
		self.weighted_add_layers.append(gen_weighted_add)
		output = gen_weighted_add([upsampled_old_to_rgb, new_block_output])
		output = layers.Activation(tf.keras.activations.tanh, name='to_skip_tanh')(output)

		self.generator = tf.keras.Model(inputs=input_layer, outputs=output)

	def update_discriminator(self):
		self.image_size = self.generator.output_shape[1]
		n_channels = self.generator.output_shape[-1]
		input_shape = [self.image_size, self.image_size, n_channels+self.n_classes]
		input_layer = layers.Input(shape=input_shape, name='to_skip_input')

		old_discriminator = []
		for layer in self.discriminator.layers:
			if 'to_skip' not in layer.name and layer.name != 'from_rgb':
				old_discriminator.append(layer)
			elif layer.name == 'from_rgb':
				old_from_rgb_output = layers.AveragePooling2D(name='to_skip_avg_pool')(input_layer)
				old_from_rgb_output = layer(old_from_rgb_output)
				layer._name = 'to_skip_old_from_rgb'
		
		new_block_output = self.disc_input_layer(
			self.n_filters//self.factor, name='from_rgb')(input_layer)
		new_block_output = self.building_block(
			[self.n_filters//self.factor, 2*self.n_filters//self.factor],
			**self.block_kwargs)(new_block_output)
		new_block_output = tf.keras.layers.AveragePooling2D()(new_block_output)

		disc_weighted_add = WeightedAdd(alpha=0.0, name='to_skip_add')
		self.weighted_add_layers.append(disc_weighted_add)
		output = disc_weighted_add([old_from_rgb_output, new_block_output])

		for layer in old_discriminator:
			output = layer(output)

		self.discriminator = tf.keras.Model(inputs=input_layer, outputs=output)

	def train(self, train_data, epochs, n_epochs_grow, path=None, compare=True, n=4, model_graph_path=None,
		callbacks=None
		):
		fade_in_mode = False
		finished = False
		delta = 1/(n_epochs_grow*len(train_data))
		el_spec = train_data.element_spec[0] if self.conditional else train_data.element_spec
		original_size = el_spec.shape[1]
		logs = {}

		if callbacks is None:
			callbacks = tf.keras.callbacks.CallbackList()

		callbacks.on_train_begin(logs=logs)

		for epoch in range(epochs):
			print(fr'Epoch {epoch+1} / {epochs}')
			callbacks.on_epoch_begin(epoch, logs=logs)
			
			if (epoch+1)%n_epochs_grow == 0:
				if (epoch+1)%(2*n_epochs_grow) != 0 and not finished:
					fade_in_mode = True
					print('fade in mode activated')

					self.weighted_add_layers = []
					self.update_generator()
					self.update_discriminator()
					self.factor *= 2

					if model_graph_path is not None:
						tf.keras.utils.plot_model(
							self.generator, show_shapes=True, expand_nested=True,
							to_file=fr'{model_graph_path}/generator_{epoch}.jpg')
						tf.keras.utils.plot_model(
							self.discriminator, show_shapes=True, expand_nested=True,
							to_file=fr'{model_graph_path}/discriminator_{epoch}.jpg')
					
					finished = self.image_size == original_size
				else:
					fade_in_mode = False
					print('stabilizing mode activated')

			with tqdm(total=len(train_data)) as progress:
				for step, data in enumerate(train_data):
					callbacks.on_batch_begin(step, logs=logs)
        			callbacks.on_train_batch_begin(step, logs=logs)

					if fade_in_mode:
						for layer in self.weighted_add_layers:
							if layer.alpha + delta <= 1:
								layer.alpha.assign_add(delta)
							else:
								layer.alpha.assign(1.0)

					if self.conditional:
						data, labels = data

					data = tf.cond(
						original_size == self.image_size,
						lambda: data,
						lambda: tf.keras.layers.Resizing(
							self.image_size, self.image_size,
							crop_to_aspect_ratio=True)(data)
					)

					if self.conditional:
						data = (data, labels)

					self.batch_size = tf.shape(data[0] if self.conditional else data)[0]
					noise = tf.random.normal([self.batch_size, self.latent_dim])

					with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
						real_output, fake_output, gp = self((data, noise), training=True)
						gen_loss = self.generator_loss(fake_output)
						disc_loss = self.discriminator_loss(real_output, fake_output)
						disc_loss += self.gp_weight*gp + self.drift*tf.reduce_mean(tf.square(real_output))

					gen_grad = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
					disc_grad = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

					self.gen_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables))
					self.disc_optimizer.apply_gradients(zip(disc_grad, self.discriminator.trainable_variables))

					self.gen_loss_tracker.update_state(gen_loss)
					self.disc_loss_tracker.update_state(disc_loss)

					progress.update(1)

					# if step == 0:
					# 	data_to_plot = data

					callbacks.on_train_batch_end(step, logs=logs)
        			callbacks.on_batch_end(step, logs=logs)

			# if path is not None:
			# 	image_path = os.path.join(path, str(epoch + 1))
			# 	labels = None
			# 	if self.conditional:
			# 		labels = tf.keras.utils.to_categorical(
			# 			np.random.choice(self.n_classes, size=n),
			# 			num_classes=self.n_classes
			# 		)

			# 	if compare:
			# 		self.generate_and_compare_samples(
			# 			real_data=data_to_plot, n=n, out_path=image_path, labels=labels)
			# 	else:
			# 		self.model.generate_new_samples(
			# 			data=self.data, n=self.n, out_path=image_path, labels=self.labels)
			
			gen_loss_final = self.gen_loss_tracker.result()
			disc_loss_final = self.disc_loss_tracker.result()
			print(
				fr'generator loss: {gen_loss_final :.4f}, ',
				fr'discriminator loss: {disc_loss_final :.4f}'
			)
			print('')

			self.gen_loss_tracker.reset_states()
			self.disc_loss_tracker.reset_states()

			callbacks.on_epoch_end(epoch, logs=logs)

		callbacks.on_train_end(logs=logs)

	def generate_and_compare_samples(self, real_data, gen_data=None, n=4, out_path=None, labels=None):
		if gen_data is None:
			gen_data = tf.random.normal([n, self.latent_dim])
		else:
			n = tf.shape(gen_data)[0].numpy()

		if self.conditional:
			gen_data = tf.concat([gen_data, labels], axis=1)
			real_data, real_labels = real_data

			string_labels = np.array(self.classes)[np.argmax(labels, axis=1)]
			real_string_labels = np.array(self.classes)[np.argmax(real_labels, axis=1)]

		new_samples = (self.generator.predict(gen_data, verbose=0) + 1) / 2
		idx = np.random.choice(tf.shape(real_data)[0], size=n, replace=False)

		fig, axes = plt.subplots(2, n, figsize=(16,10))
		axes = axes.flatten()
		for i in range(n):
			real_sample = (real_data.numpy()[idx[i], :].squeeze() + 1) / 2
			cmap = 'gray' if np.ndim(real_sample) == 2 else None
			axes[i].imshow(real_sample, cmap=cmap)
			axes[i].axis('off')

			gen_sample = new_samples[i, :].squeeze()
			axes[i+n].imshow(gen_sample, cmap=cmap)
			axes[i+n].axis('off')

			if self.conditional:
				axes[i].set_title(real_string_labels[i])
				axes[i+n].set_title(string_labels[i])

		if out_path is not None:
			plt.savefig(out_path, bbox_inches='tight')
			plt.close(fig)