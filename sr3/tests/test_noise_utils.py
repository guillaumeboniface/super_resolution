import unittest
import numpy as np
from sr3.noise_utils import sample_noise_schedule

class TestNoiseScheduleSample(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(3)
        return super().setUp()
    
    def test_sampler(self) -> None:
        schedule = list(range(5))
        sample = sample_noise_schedule(schedule, 3)
        self.assertEqual(sample.shape, (3,))
        self.assertAlmostEqual(sum(sample - np.array([2.83994904, 0.83994904, 1.83994904])), 0)

if __name__ == '__main__':
    unittest.main()