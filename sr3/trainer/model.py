import tensorflow as tf
import tensorflow_addons as tfa
from sr3.utils import WarmUpSchedule
import sr3.trainer.components as components 
from collections.abc import Iterable

def create_model(
        img_shape: Iterable,
        batch_size: int,
        channel_dim: int = 128,
        channel_ramp_multiplier: Iterable = (1, 2, 4, 8, 8),
        attention_resolution: Iterable = (8,),
        out_channels: int = 3,
        num_resblock: int = 3,
        dropout: float = 0.2,
        use_deep_blocks: bool = False,
        resample_with_conv: bool = False) -> tf.keras.Model:

    if use_deep_blocks:
        upblock = components.up_deep_resblock
        block = components.deep_resblock
    else:
        upblock = components.up_resblock
        block = components.resblock

    num_resolutions = len(channel_ramp_multiplier)

    squashed_input = tf.keras.Input(shape=(3,) + img_shape, batch_size=batch_size) # inputs had to be stacked to go through the data pipeline
    context_img = tf.gather(squashed_input, 0, axis=1)
    noisy_img = tf.gather(squashed_input, 1, axis=1)
    gamma = tf.gather(squashed_input, 2, axis=1)
    gamma = tf.reduce_mean(gamma, axis=(1, 2, 3)) # gamma had to be broadcasted to be stacked, this is the reverse
    gamma = tf.expand_dims(gamma, 1)
    combined_images = tf.keras.layers.Concatenate(axis=-1)([noisy_img, context_img])

    noise_level_embedding = components.NoiseEmbedding(channel_dim)(gamma)
    noise_level_mlp = tf.keras.models.Sequential()
    noise_level_mlp.add(tf.keras.layers.Dense(channel_dim * 4, activation="swish"))
    noise_level_mlp.add(tf.keras.layers.Dense(channel_dim))
    noise_level_embedding = noise_level_mlp(noise_level_embedding)
    
    x = combined_images
    x = tf.keras.layers.Conv2D(channel_dim, 3, padding='same')(x)
    skip_connections = [x]
    for i, multiplier in enumerate(channel_ramp_multiplier):
        for j in range(num_resblock):
            x = block(x, noise_level_embedding, channel_dim * multiplier, dropout=dropout)
            if x.shape[1] in attention_resolution:
                x = components.attention_block(x)
            skip_connections.append(x)
        if i != num_resolutions - 1:
            x = components.downsample(x, use_conv=resample_with_conv)
            skip_connections.append(x)

    x = block(x, noise_level_embedding, dropout=dropout)
    x = components.attention_block(x)
    x = block(x, noise_level_embedding, dropout=dropout)

    for i, multiplier in reversed(list(enumerate(channel_ramp_multiplier))):
        for j in range(num_resblock + 1):
            x = upblock(x, skip_connections.pop(), noise_level_embedding, channel_dim * multiplier, dropout=dropout)
            if x.shape[1] in attention_resolution:
                x = components.attention_block(x)
        if i != 0:
            x = components.upsample(x, use_conv=resample_with_conv)

    assert(len(skip_connections) == 0)

    x = tfa.layers.GroupNormalization(groups=32, axis=-1)(x)
    x = tf.keras.activations.swish(x)
    outputs = tf.keras.layers.Conv2D(out_channels, 3, padding='same')(x)

    model = tf.keras.Model(inputs=squashed_input, outputs=outputs)
    return model

custom_objects = {
    'WarmUpSchedule': WarmUpSchedule,
    'ConditionalInstanceNormalization': components.ConditionalInstanceNormalization,
    'AttentionVectorLayer': components.AttentionVectorLayer,
    'NoiseEmbedding': components.NoiseEmbedding
}