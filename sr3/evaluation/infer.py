import tensorflow as tf
from collections.abc import Iterable

from tensorflow.python.ops.image_ops_impl import ResizeMethod
from sr3.noise_utils import noise_schedule, generate_noisy_image_batch
from sr3.dataset import IMAGE_SHAPE
from sr3.trainer.model import custom_objects
import fire

def infer(model: tf.keras.models.Model, images: tf.Tensor, alpha_noise_schedule: Iterable) -> tf.Tensor:
    alphas = tf.cast(alpha_noise_schedule, tf.float32)
    gammas = tf.cast(tf.math.cumprod(alpha_noise_schedule), tf.float32)
    alphas = tf.reverse(alphas, (0,))
    gammas = tf.reverse(gammas, (0,))
    batch_size = images.shape[0]
    context_images = images
    target_images, noise = generate_noisy_image_batch(images, gammas[:1])
    for i in range(alphas.shape[0]):
        gamma_b = tf.broadcast_to(tf.reshape(gammas[i], (batch_size, 1, 1, 1)), (batch_size,) + IMAGE_SHAPE)
        input = tf.stack([context_images, target_images, gamma_b], axis=1)
        z = tf.math.sqrt(1 - alphas[i]) * tf.random.normal(images.shape, stddev=1.) if i < alphas.shape[0] - 1 else tf.zeros(images.shape)
        target_images = (target_images - (1 - alphas[i]) / tf.math.sqrt(1 - gammas[i]) * model(input)) / tf.math.sqrt(alphas[i]) + z
            
    return target_images

def infer_from_file(
        model_path: str,
        src: str,
        dest: str,
        noise_schedule_shape: str = 'linear',
        noise_schedule_start: float = 0.9999999,
        noise_schedule_end: float = 0.9,
        noise_schedule_steps: int = 100) -> None:
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    alphas = noise_schedule(noise_schedule_shape, noise_schedule_start, noise_schedule_end, noise_schedule_steps)
    image = tf.image.convert_image_dtype(tf.io.decode_jpeg(tf.io.read_file(src)), tf.float32)
    h, w, c = image.shape
    image = tf.expand_dims(image, 0)
    image = tf.image.resize(image, (h * 8, w * 8), method=ResizeMethod.NEAREST_NEIGHBOR)
    result_img = tf.squeeze(infer(model, image, alphas))
    result_img = tf.image.convert_image_dtype(result_img, tf.uint8)
    tf.io.write_file(dest, tf.io.encode_jpeg(result_img))
    return

if __name__ == '__main__':
  fire.Fire(infer_from_file)