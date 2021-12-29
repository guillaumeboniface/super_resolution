import unittest
import tensorflow as tf
from sr3.noise_utils import sample_noise_schedule, noise_schedule, generate_noisy_image, generate_noisy_image_minus_one_batch, generate_noisy_image_batch

class TestNoiseScheduleGenerator(unittest.TestCase):

    def test_jsd(self) -> None:
        alphas = noise_schedule('jsd', 0.9, 0.9, 25)
        self.assertAlmostEqual(alphas.numpy()[0], 1. / 25.)
        self.assertAlmostEqual(alphas.numpy()[24], 1.)
        self.assertAlmostEqual(alphas.numpy()[12], 1. / 13.)

class TestNoiseScheduleSample(unittest.TestCase):

    def setUp(self) -> None:
        tf.experimental.numpy.random.seed(3)
        return super().setUp()
    
    def test_sampler(self) -> None:
        schedule = tf.convert_to_tensor(list(range(5)), dtype=tf.float64)
        alpha_sample, gamma_sample = sample_noise_schedule(schedule, 3)
        self.assertEqual(alpha_sample.shape, (3,))
        self.assertAlmostEqual(tf.reduce_sum(alpha_sample - tf.convert_to_tensor([0.4434243, 0.00790431, 3.4255726])).numpy(), 0)


class TestBatchNoiseImageGeneration(unittest.TestCase):
    
    def test_noisy_image_generator(self) -> None:
        batch_size = 10
        images = tf.ones((batch_size, 128, 128, 3))
        schedule = tf.convert_to_tensor(list(range(5)), dtype=tf.float64)
        alpha_sample, gamma_sample = sample_noise_schedule(schedule, batch_size)
        noisy_images, noise = generate_noisy_image_batch(images, alpha_sample)
        self.assertEqual(images.shape, noisy_images.shape)


if __name__ == '__main__':
    unittest.main()