import tensorflow as tf
from sr3.trainer.components import *
from collections.abc import Iterable

def create_model(
        img_shape: Iterable[int] = (128, 128, 3),
        batch_size: int = 32,
        channel_dim: int = 128,
        channel_ramp_multiplier: Iterable[int] = (1, 2, 4, 8, 8),
        attention_resolution: Iterable[int] = (8,),
        out_channels: int = 3,
        num_resblock: int = 3,
        dropout: float = 0.2,
        use_deep_blocks: bool = False,
        resample_with_conv: bool = False) -> tf.keras.Model:

    if use_deep_blocks:
        upblock = up_deep_resblock
        block = deep_resblock
    else:
        upblock = up_resblock
        block = resblock

    num_resolutions = len(channel_ramp_multiplier)

    squashed_input = tf.keras.Input(type_spec=tf.TensorSpec((3,) + (batch_size,) + img_shape)) # inputs had to be stacked to go through the data pipeline
    context_img = squashed_input[0]
    noisy_img = squashed_input[1]
    gamma = squashed_input[2]
    gamma = tf.reduce_mean(gamma, axis=(1, 2, 3)) # gamma had to be broadcasted to be stacked, this is the reverse
    gamma = tf.reshape(gamma, (batch_size, 1))
    combined_images = tf.keras.layers.Concatenate(axis=-1)([noisy_img, context_img])
    
    x = combined_images
    x = tf.keras.layers.Conv2D(channel_dim, 3, padding='same')(x)
    skip_connections = [x]
    for i, multiplier in enumerate(channel_ramp_multiplier):
        for j in range(num_resblock):
            x = block(x, gamma, channel_dim * multiplier, dropout=dropout)
            if x.shape[1] in attention_resolution:
                x = attention_block(x, gamma)
            skip_connections.append(x)
        if i != num_resolutions - 1:
            x = downsample(x, use_conv=resample_with_conv)
            skip_connections.append(x)

    x = block(x, gamma, dropout=dropout)
    x = attention_block(x, gamma)
    x = block(x, gamma, dropout=dropout)

    for i, multiplier in reversed(list(enumerate(channel_ramp_multiplier))):
        for j in range(num_resblock + 1):
            x = upblock(x, skip_connections.pop(), gamma, channel_dim * multiplier, dropout=dropout)
            if x.shape[1] in attention_resolution:
                x = attention_block(x, gamma)
        if i != 0:
            x = upsample(x, use_conv=resample_with_conv)

    assert(len(skip_connections) == 0)

    x = ConditionalInstanceNormalization()([x, gamma])
    x = tf.keras.activations.swish(x)
    outputs = tf.keras.layers.Conv2D(out_channels, 3, padding='same')(x)

    model = tf.keras.Model(inputs=squashed_input, outputs=outputs)
    return model