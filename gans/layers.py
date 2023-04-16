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


class CreateLearnableConstant(tf.keras.layers.Layer):
    def __init__(self, size, channels, name=None, **kwargs):
        super(CreateLearnableConstant, self).__init__(**kwargs, name=name)
        self.size = size
        self.channels = channels

    def build(self, input_shape):
        self.const_input = self.add_weight(
            'const_input',
            shape=[1, self.size, self.size, self.channels],
            initializer='ones',
            trainable=True,
            dtype=tf.float32
        )
        
    def call(self, inputs):
        target_shape = [tf.shape(inputs)[0], 1, 1, 1]
        return tf.tile(self.const_input, target_shape)


class AddScaledNoise(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(AddScaledNoise, self).__init__(**kwargs, name=name)

    def build(self, input_shape):
        self.channel_weights = self.add_weight(
            'channel_weights',
            shape=[1, 1, 1, input_shape[-1]],
            initializer='zeros',
            trainable=True,
            dtype=tf.float32
        )
        
    def call(self, inputs):
        noise = tf.random.normal(
            [tf.shape(inputs)[0], inputs.shape[1], inputs.shape[2], 1],
            dtype=inputs.dtype
        )
        return inputs + noise * self.channel_weights


class StyleAdaIN(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(StyleAdaIN, self).__init__(**kwargs, name=name)

    def build(self, input_shape):
        self.affine_transform = tf.keras.layers.Dense(
            2*input_shape[0][-1], kernel_initializer='he_uniform')
        
    def call(self, inputs):
        x, latents = inputs[0], inputs[1]
        y = self.affine_transform(latents)
        y_s, y_b = tf.split(y, num_or_size_splits=2, axis=1)

        x_mean = tf.reduce_mean(x, axis=[1,2], keepdims=True)
        x_std = tf.math.reduce_std(x, axis=[1,2], keepdims=True)
        x_norm = (x - x_mean) / (x_std + 1e-8)

        y_s = tf.reshape(y_s, [tf.shape(y_s)[0], 1, 1, y_s.shape[1]])
        y_b = tf.reshape(y_b, [tf.shape(y_b)[0], 1, 1, y_b.shape[1]])
        
        return y_s * x_norm + y_b