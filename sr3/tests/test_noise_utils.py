import unittest
import tensorflow as tf
from sr3.noise_utils import *

class TestNoiseScheduleSample(unittest.TestCase):

    def setUp(self) -> None:
        tf.experimental.numpy.random.seed(3)
        return super().setUp()
    
    def test_sampler(self) -> None:
        schedule = tf.convert_to_tensor(list(range(5)), dtype=tf.float64)
        alpha_sample, gamma_sample, gamma_minus_one_sample = sample_noise_schedule(schedule, 3)
        self.assertEqual(alpha_sample.shape, (3,))
        self.assertAlmostEqual(tf.reduce_sum(alpha_sample - tf.convert_to_tensor([0.4434243, 0.00790431, 3.4255726])).numpy(), 0)


class TestBatchNoiseImageGeneration(unittest.TestCase):
    
    def test_noisy_image_generator(self) -> None:
        batch_size = 10
        images = tf.ones((batch_size, 128, 128, 3))
        schedule = tf.convert_to_tensor(list(range(5)), dtype=tf.float64)
        alpha_sample, gamma_sample, gamma_minus_one_sample = sample_noise_schedule(schedule, batch_size)
        noisy_images = generate_noisy_image_batch(images, alpha_sample)
        self.assertEqual(images.shape, noisy_images.shape)

    def test_minus_one_image_generator(self) -> None:
        img = tf.io.decode_jpeg(tf.io.read_file("./resources/00002.jpg"))
        img = tf.image.convert_image_dtype(img, tf.float32)
        gamma_schedule = tf.cast(noise_schedule('quad', 1., 0.8, 5, gammas=True), tf.float32)
        alpha_schedule = tf.cast(noise_schedule('quad', 1., 0.8, 5), tf.float32)
        t = 2
        noisy_img_t = generate_noisy_image(img, gamma_schedule[t])
        noisy_img_minus_one = generate_noisy_image(img, gamma_schedule[t-1])
        syn_minus_one = generate_noisy_image_minus_one_batch(
        tf.convert_to_tensor([img]),
        tf.convert_to_tensor([noisy_img_t]),
        tf.convert_to_tensor([gamma_schedule[t]]),
        tf.convert_to_tensor([gamma_schedule[t - 1]]),
        tf.convert_to_tensor([alpha_schedule[t]]))
        syn_minus_one = tf.squeeze(syn_minus_one)
        self.assertLess(tf.math.abs(tf.math.reduce_std(syn_minus_one) - tf.math.reduce_std(noisy_img_minus_one)).numpy(), 0.01)
        self.assertLess(tf.math.abs(tf.reduce_mean(syn_minus_one) - tf.reduce_mean(noisy_img_minus_one)).numpy(), 0.01)
        self.assertGreater(tf.math.abs(tf.reduce_mean(syn_minus_one) - tf.reduce_mean(noisy_img_t)).numpy(), 0.03)


if __name__ == '__main__':
    unittest.main()