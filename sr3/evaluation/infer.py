import tensorflow as tf
from collections.abc import Iterable

from tensorflow.python.ops.image_ops_impl import ResizeMethod
from sr3.trainer.components import *
from sr3.utils import WarmUpSchedule
from sr3.noise_utils import noise_schedule
from sr3.datasets.celebhq import *
import fire

def infer(model: tf.keras.models.Model, images: tf.Tensor, alpha_noise_schedule: Iterable) -> tf.Tensor:
    alphas = tf.cast(alpha_noise_schedule, tf.float32)
    gammas = tf.cast(tf.math.cumprod(alpha_noise_schedule), tf.float32)
    batch_size = images.shape[0]
    context_images = images
    target_images = tf.random.normal(context_images.shape, stddev=1.)
    for i in range(alphas.shape[0]):
        gamma_b = tf.broadcast_to(tf.reshape(gammas[i], (batch_size, 1, 1, 1)), (batch_size,) + IMAGE_SHAPE)
        input = tf.stack([context_images, target_images, gamma_b], axis=1)
        target_images = (target_images - (1 - alphas[i]) / tf.math.sqrt(1 - gammas[i]) * model(input)) / tf.math.sqrt(alphas[i]) + \
            tf.math.sqrt(1 - alphas[i]) * tf.random.normal(images.shape, stddev=1.)
    return target_images

def infer_from_file(
        model_path: str,
        src: str,
        dest: str,
        noise_schedule_shape: str = 'quad',
        noise_schedule_start: float = 1.,
        noise_schedule_end: float = 0.95,
        noise_schedule_steps: int = 100) -> None:
    model = tf.keras.models.load_model(model_path, custom_objects={
        'WarmUpSchedule': WarmUpSchedule,
        'ConditionalInstanceNormalization': ConditionalInstanceNormalization,
        'AttentionVectorLayer': AttentionVectorLayer
    })
    alphas = noise_schedule(noise_schedule_shape, noise_schedule_start, noise_schedule_end, noise_schedule_steps)
    image = tf.cast(tf.io.decode_jpeg(tf.io.read_file(src)), tf.float32)
    h, w, c = image.shape
    image = tf.expand_dims(image, 0)
    image = tf.image.resize(image, (h * 8, w * 8), method=ResizeMethod.NEAREST_NEIGHBOR)
    result_img = tf.squeeze(infer(model, image, alphas))
    result_img = tf.cast(result_img, tf.uint8)
    tf.io.write_file(dest, tf.io.encode_jpeg(result_img))
    return

if __name__ == '__main__':
  fire.Fire(infer_from_file)