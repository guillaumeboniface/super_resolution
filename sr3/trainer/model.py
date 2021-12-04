import tensorflow as tf
from sr3.trainer.components import *

def create_model(
        img_shape=(128, 128, 3),
        batch_size=32,
        channel_dim=128,
        channel_ramp_multiplier=(1, 2, 4, 8, 8),
        attention_resolution=(8,),
        out_channels=3,
        num_resblock=3,
        use_deep_blocks=True,
        resample_with_conv=False):

    if use_deep_blocks:
        upblock = up_deep_resblock
        block = deep_resblock
    else:
        upblock = up_resblock
        block = resblock

    num_resolutions = len(channel_ramp_multiplier)

    context_img = tf.keras.Input(shape=img_shape, batch_size=batch_size)
    noisy_img = tf.keras.Input(shape=img_shape, batch_size=batch_size)
    gamma = tf.keras.Input(shape=(1,), batch_size=batch_size)
    combined_images = tf.keras.layers.Concatenate(axis=-1)([noisy_img, context_img])
    
    x = combined_images
    x = tf.keras.layers.Conv2D(channel_dim, 3, padding='same')(x)
    skip_connections = [x]
    for i, multiplier in enumerate(channel_ramp_multiplier):
        for j in range(num_resblock):
            x = block(x, gamma, channel_dim * multiplier)
            if x.shape[1] in attention_resolution:
                x = attention_block(x, gamma)
            skip_connections.append(x)
        if i != num_resolutions - 1:
            x = downsample(x, use_conv=resample_with_conv)
            skip_connections.append(x)

    x = block(x, gamma)
    x = attention_block(x, gamma)
    x = block(x, gamma)

    for i, multiplier in reversed(list(enumerate(channel_ramp_multiplier))):
        for j in range(num_resblock + 1):
            x = upblock(x, skip_connections.pop(), gamma, channel_dim * multiplier)
            if x.shape[1] in attention_resolution:
                x = attention_block(x, gamma)
        if i != 0:
            x = upsample(x, use_conv=resample_with_conv)

    x = ConditionalInstanceNormalization()([x, gamma])
    x = tf.keras.activations.swish(x)
    outputs = tf.keras.layers.Conv2D(out_channels, 3, padding='same')(x)

    model = tf.keras.Model(inputs=[context_img, noisy_img, gamma], outputs=outputs)
    return model