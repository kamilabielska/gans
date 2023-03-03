import tensorflow as tf


class GaussianNoiseAnnealing(tf.keras.layers.GaussianNoise):
	def __init__(self, stddev, **kwargs):
		super(GaussianNoiseAnnealing, self).__init__(stddev=stddev, **kwargs)
		self.stddev = tf.Variable(initial_value=stddev, trainable=False)


class WeightedAdd(tf.keras.layers.Layer):
	def __init__(self, alpha, name=None, **kwargs):
		super(WeightedAdd, self).__init__(**kwargs, name=name)
		self.alpha = tf.Variable(initial_value=alpha, trainable=False)

	def call(self, inputs):
		return (1-self.alpha)*inputs[0] + self.alpha*inputs[1]


class MinibatchSTD(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		super(MinibatchSTD, self).__init__(**kwargs)

	def call(self, inputs):
		shape = tf.shape(inputs)
		mean_std = tf.reduce_mean(tf.math.reduce_std(inputs, axis=[0, -1]))
		mean_std_map = tf.fill([shape[0], shape[1], shape[2], 1], mean_std)
		return tf.concat([inputs, mean_std_map], axis=-1)

	def compute_output_shape(self, input_shape):
		input_shape = list(input_shape)
		input_shape[-1] += 1
		return tuple(input_shape)


class PixelNorm(tf.keras.layers.Layer):
	def __init__(self, name=None, **kwargs):
		super(PixelNorm, self).__init__(**kwargs, name=name)

	def call(self, inputs):
		mean_square_inputs = tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True)
		return inputs / tf.sqrt(mean_square_inputs + 1e-8)