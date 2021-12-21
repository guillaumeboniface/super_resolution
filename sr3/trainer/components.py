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

def attention_block(inputs: Iterable) -> tf.Tensor:
    """
    Implementing self-attention block, as mentioned in
    https://arxiv.org/pdf/1809.11096.pdf
    """

    x, noise_emb = inputs

    x = ConditionalInstanceNormalization()([x, noise_emb])

    q = AttentionVectorLayer()(x)
    v = AttentionVectorLayer()(x)
    k = AttentionVectorLayer()(x)

    h = tf.keras.layers.Attention()([q, v, k])

    return x + h

def deep_resblock(
        x: tf.Tensor,
        noise_embedding: tf.Tensor,
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

    track_b = ConditionalInstanceNormalization()([x, noise_embedding])
    track_b = tf.keras.activations.swish(track_b)
    track_b = tf.keras.layers.Conv2D(n_out_channels, 1, padding='same')(track_b)
    track_b = ConditionalInstanceNormalization()([track_b, noise_embedding])
    track_b = tf.keras.activations.swish(track_b)
    track_b = tf.keras.layers.Conv2D(n_out_channels, 3, padding='same')(track_b)
    track_b = tf.keras.layers.Dropout(dropout)(track_b)
    track_b = ConditionalInstanceNormalization()([track_b, noise_embedding])
    track_b = tf.keras.activations.swish(track_b)
    track_b = tf.keras.layers.Conv2D(n_out_channels, 3, padding='same')(track_b)
    track_b = ConditionalInstanceNormalization()([track_b, noise_embedding])
    track_b = tf.keras.activations.swish(track_b)
    track_b = tf.keras.layers.Conv2D(n_out_channels, 1, padding='same')(track_b)

    return track_a + track_b

def resblock(
        x: tf.Tensor,
        noise_embedding: tf.Tensor,
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

    track_b = ConditionalInstanceNormalization()([x, noise_embedding])
    track_b = tf.keras.activations.swish(track_b)
    track_b = tf.keras.layers.Conv2D(n_out_channels, 3, padding='same')(track_b)
    track_b = tf.keras.layers.Dropout(dropout)(track_b)
    track_b = ConditionalInstanceNormalization()([track_b, noise_embedding])
    track_b = tf.keras.activations.swish(track_b)
    track_b = tf.keras.layers.Conv2D(n_out_channels, 3, padding='same')(track_b)

    return track_a + track_b

def up_deep_resblock(x: tf.Tensor, skip_x: tf.Tensor, noise_embedding: tf.Tensor, n_out_channels: bool = None,
        dropout: float = 0.) -> tf.Tensor:
    if not n_out_channels:
        n_out_channels = x.shape[-1]

    skip_x = skip_x / tf.math.sqrt(2.) # downscale the skip connection by 1 / sqrt(2)
    x = tf.keras.layers.Concatenate(axis=-1)([x, skip_x])
    
    return deep_resblock(x, noise_embedding, n_out_channels=n_out_channels, dropout=dropout)

def up_resblock(x: tf.Tensor, skip_x: tf.Tensor, noise_embedding: tf.Tensor, n_out_channels: bool = None,
        dropout: float = 0.) -> tf.Tensor:
    if not n_out_channels:
        n_out_channels = x.shape[-1]

    skip_x = skip_x / tf.math.sqrt(2.)
    x = tf.keras.layers.Concatenate(axis=-1)([x, skip_x])
    
    return resblock(x, noise_embedding, n_out_channels=n_out_channels, dropout=dropout)

class AttentionVectorLayer(tf.keras.layers.Layer):
    """
    Building the query, key or value for self-attention
    from the feature map
    """
    def __init__(self, **kwargs):
        super(AttentionVectorLayer, self).__init__(**kwargs)

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

    def get_config(self) -> dict:
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ConditionalInstanceNormalization(tf.keras.layers.Layer):
    """
    The goal of conditional instance normalization is to make the
    model aware of the amount of noise to be removed (i.e how many
    steps in the noise diffusion process are being considered). The
    implementation was informed by the appendix in
    https://arxiv.org/pdf/1907.05600.pdf and implementation at
    https://github.com/ermongroup/ncsn/blob/master/models/cond_refinenet_dilated.py
    """
    def __init__(self, **kwargs):
        super(ConditionalInstanceNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.batch_size = input_shape[0][0]
        self.height = input_shape[0][1]
        self.width = input_shape[0][2]
        self.n_channels = input_shape[0][3]
        self.embedding_dim = input_shape[1][1]
        self.w1 = self.add_weight(
            shape=(self.embedding_dim, self.n_channels),
            initializer=tf.keras.initializers.RandomNormal(mean=1., stddev=0.02),
            trainable=True,
            name='conditional_w1'
        )
        self.b = self.add_weight(
            shape=(self.embedding_dim, self.n_channels), initializer="zero", trainable=True,
            name='conditional_b'
        )
        self.w2 = self.add_weight(
            shape=(self.embedding_dim, self.n_channels),
            initializer=tf.keras.initializers.RandomNormal(mean=1., stddev=0.02),
            trainable=True,
            name='conditional_w2'
        )

    def call(self, inputs: Iterable) -> tf.Tensor:
        x, noise_embedding = inputs
        feature_map_means = tf.math.reduce_mean(x, axis=(1, 2), keepdims=True)
        feature_map_std_dev = tf.math.reduce_std(x, axis=(1, 2), keepdims=True) + 1e-5
        m = tf.math.reduce_mean(feature_map_means, axis=-1, keepdims=True)
        v = tf.math.reduce_std(feature_map_means, axis=-1, keepdims=True) + 1e-5
        gamma = tf.expand_dims(tf.expand_dims(tf.tensordot(noise_embedding, self.w1, 1), 1), 1)
        beta = tf.expand_dims(tf.expand_dims(tf.tensordot(noise_embedding, self.b, 1), 1), 1)
        alpha = tf.expand_dims(tf.expand_dims(tf.tensordot(noise_embedding, self.w2, 1), 1), 1)
        instance_norm = (x - feature_map_means) / feature_map_std_dev
        x = gamma * instance_norm + beta + alpha * (feature_map_means - m) / v
        return x

    def get_config(self) -> dict:
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class NoiseEmbedding(tf.keras.layers.Layer):
    """
    Implementation of noise embedding taken from Wavegrad
    implementation at https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
    """
    def __init__(self, dim, **kwargs):
        super(NoiseEmbedding, self).__init__(**kwargs)
        self.dim = dim

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        assert(len(inputs.shape) == 2)
        half_dim = self.dim // 2
        encoding = tf.range(half_dim, dtype=tf.float32) * tf.math.log(1e4) / half_dim
        encoding = tf.math.exp(encoding * -1)
        encoding = inputs * tf.expand_dims(encoding, 0)
        encoding = tf.concat([tf.math.sin(encoding), tf.math.cos(encoding)], -1)
        assert(encoding.shape == (inputs.shape[0], self.dim))
        return encoding

    def get_config(self) -> dict:
        return {
            'dim': self.dim
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

