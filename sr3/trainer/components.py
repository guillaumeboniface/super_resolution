import tensorflow as tf
from collections.abc import Iterable

def upsample(x: tf.Tensor, use_conv: bool = False) -> tf.Tensor:
    batch_size, height, width, channels = x.shape
    x = tf.keras.layers.UpSampling2D(interpolation='nearest')(x)
    assert(x.shape == [batch_size, height * 2, width * 2, channels])
    if use_conv:
        x = tf.keras.layers.Conv2D(channels, 3, padding='same')(x)
        assert(x.shape == [batch_size, height * 2, width * 2, channels])
    return x

def downsample(x: tf.Tensor, use_conv: bool = False):
    batch_size, height, width, channels = x.shape
    if use_conv:
        x = tf.keras.layers.Conv2D(channels, 3, strides=2, padding='same')(x)
    else:
        x = tf.keras.layers.AveragePooling2D(strides=2, padding='same')(x)
    assert(x.shape == [batch_size, height // 2, width // 2, channels])
    return x

def attention_block(x: tf.Tensor, gamma: tf.Tensor) -> tf.Tensor:
    """
    Implementing self-attention block, as mentioned in
    https://arxiv.org/pdf/1809.11096.pdf
    """
    
    h = ConditionalInstanceNormalization()([x, gamma])

    q = AttentionVectorLayer()(h)
    v = AttentionVectorLayer()(h)
    k = AttentionVectorLayer()(h)

    h = tf.keras.layers.Attention()([q, v, k])

    return x + h

def deep_resblock(
        x: tf.Tensor,
        gamma: tf.Tensor,
        n_out_channels: bool = None,
        dropout: float = 0.) -> tf.Tensor:
    """
    Reuse the resblock definitions from https://arxiv.org/pdf/1809.11096.pdf
    Slight tweak to use ConditionalInstanceNormalization and pre-activation
    as described in https://arxiv.org/pdf/1907.05600.pdf
    """
    if not n_out_channels:
        n_out_channels = x.shape[-1]

    track_a = tf.keras.layers.Conv2D(n_out_channels, 1, padding='same')(x)

    track_b = ConditionalInstanceNormalization()([x, gamma])
    track_b = tf.keras.activations.swish(track_b)
    track_b = tf.keras.layers.Conv2D(n_out_channels, 1, padding='same')(track_b)
    track_b = ConditionalInstanceNormalization()([track_b, gamma])
    track_b = tf.keras.activations.swish(track_b)
    track_b = tf.keras.layers.Conv2D(n_out_channels, 3, padding='same')(track_b)
    track_b = tf.keras.layers.Dropout(dropout)(track_b)
    track_b = ConditionalInstanceNormalization()([track_b, gamma])
    track_b = tf.keras.activations.swish(track_b)
    track_b = tf.keras.layers.Conv2D(n_out_channels, 3, padding='same')(track_b)
    track_b = ConditionalInstanceNormalization()([track_b, gamma])
    track_b = tf.keras.activations.swish(track_b)
    track_b = tf.keras.layers.Conv2D(n_out_channels, 1, padding='same')(track_b)

    return track_a + track_b

def resblock(
        x: tf.Tensor,
        gamma: tf.Tensor,
        n_out_channels: bool = None,
        dropout: float = 0.) -> tf.Tensor:
    """
    Reuse the resblock definitions from https://arxiv.org/pdf/1809.11096.pdf
    Slight tweak to use ConditionalInstanceNormalization and pre-activation
    as described in https://arxiv.org/pdf/1907.05600.pdf
    """
    if not n_out_channels:
        n_out_channels = x.shape[-1]

    track_a = tf.keras.layers.Conv2D(n_out_channels, 1, padding='same')(x)

    track_b = ConditionalInstanceNormalization()([x, gamma])
    track_b = tf.keras.activations.swish(track_b)
    track_b = tf.keras.layers.Conv2D(n_out_channels, 3, padding='same')(track_b)
    track_b = tf.keras.layers.Dropout(dropout)(track_b)
    track_b = ConditionalInstanceNormalization()([track_b, gamma])
    track_b = tf.keras.activations.swish(track_b)
    track_b = tf.keras.layers.Conv2D(n_out_channels, 3, padding='same')(track_b)

    return track_a + track_b

def up_deep_resblock(x: tf.Tensor, skip_x: tf.Tensor, gamma: tf.Tensor, n_out_channels: bool = None,
        dropout: float = 0.) -> tf.Tensor:
    if not n_out_channels:
        n_out_channels = x.shape[-1]

    skip_x = skip_x * 1 / tf.math.sqrt(2.)
    x = tf.keras.layers.Concatenate(axis=-1)([x, skip_x])
    
    return deep_resblock(x, gamma, n_out_channels=n_out_channels, dropout=dropout)

def up_resblock(x: tf.Tensor, skip_x: tf.Tensor, gamma: tf.Tensor, n_out_channels: bool = None,
        dropout: float = 0.) -> tf.Tensor:
    if not n_out_channels:
        n_out_channels = x.shape[-1]

    skip_x = skip_x * 1 / tf.math.sqrt(2.)
    x = tf.keras.layers.Concatenate(axis=-1)([x, skip_x])
    
    return resblock(x, gamma, n_out_channels=n_out_channels, dropout=dropout)

class AttentionVectorLayer(tf.keras.layers.Layer):
    """
    Building the query, key or value for self-attention
    from the feature map
    """
    def __init__(self):
        super(AttentionVectorLayer, self).__init__()

    def build(self, input_shape):
        self.n_channels = input_shape[-1]
        self.w = self.add_weight(
            shape=(self.n_channels, self.n_channels),
            initializer=tf.keras.initializers.VarianceScaling(scale=1., mode='fan_avg', distribution='uniform'),
            trainable=True,
            name='attention_w'
        )
        self.b = self.add_weight(
            shape=(self.n_channels,),
            initializer='zero',
            trainable=True,
            name='attention_b'
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return tf.tensordot(x, self.w, 1) + self.b

class ConditionalInstanceNormalization(tf.keras.layers.Layer):
    """
    The goal of conditional instance normalization is to make the
    model aware of the amount of noise to be removed (i.e how many
    steps in the noise diffusion process are being considered). The
    implementation was informed by the appendix in
    https://arxiv.org/pdf/1907.05600.pdf
    """
    def __init__(self):
        super(ConditionalInstanceNormalization, self).__init__()

    def build(self, input_shape):
        self.batch_size = input_shape[0][0]
        self.height = input_shape[0][1]
        self.width = input_shape[0][2]
        self.n_channels = input_shape[0][3]
        self.w1 = self.add_weight(
            shape=(self.n_channels,),
            initializer=tf.keras.initializers.RandomNormal(mean=1., stddev=0.02),
            trainable=True,
            name='conditional_w1'
        )
        self.b = self.add_weight(
            shape=(self.n_channels,), initializer="zero", trainable=True,
            name='conditional_b'
        )
        self.w2 = self.add_weight(
            shape=(self.n_channels,),
            initializer=tf.keras.initializers.RandomNormal(mean=1., stddev=0.02),
            trainable=True,
            name='conditional_w2'
        )

    def call(self, inputs: Iterable[tf.Tensor]) -> tf.Tensor:
        x, gamma = inputs
        feature_map_means = tf.math.reduce_mean(x, axis=(1, 2), keepdims=True)
        feature_map_std_dev = tf.math.reduce_std(x, axis=(1, 2), keepdims=True)
        m = tf.math.reduce_mean(feature_map_means, axis=-1, keepdims=True)
        v = tf.math.reduce_std(feature_map_std_dev, axis=-1, keepdims=True)
        weighted_feature_map_means = tf.reshape(feature_map_means, (self.batch_size, self.n_channels)) * self.w1 * gamma
        weighted_feature_map_means = tf.reshape(weighted_feature_map_means, (self.batch_size, 1, 1, self.n_channels))
        x = (x - weighted_feature_map_means) / feature_map_std_dev + self.b + self.w2 * (feature_map_means - m) / v
        return x
