import tensorflow as tf
from tensorflow.python.eager.context import context

def create_model(img_shape=(128, 128, 3)):
    context_img = tf.keras.Input(shape=img_shape)
    # combined_images = tf.keras.layers.Concatenate(axis=-1)([noisy_img, context_img])
    outputs = tf.keras.layers.Conv2D(3, 3, padding='same', activation='swish')(context_img)
    model = tf.keras.Model(inputs=context_img, outputs=outputs)
    return model