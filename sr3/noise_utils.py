import numpy as np
import tensorflow as tf

def sample_noise_schedule(schedule, batch_size):
    n_timesteps = len(schedule)
    t = np.random.randint(n_timesteps, size=batch_size)
    start_interval = np.array(schedule)[t]
    end_interval = np.array(schedule)[t + 1]
    noise = start_interval + np.random.uniform() * (end_interval - start_interval)
    return noise

def _warmup_alpha(start, end, n_timesteps, warmup_fraction):
  alphas = end * np.ones(n_timesteps, dtype=np.float64)
  warmup_time = int(n_timesteps * warmup_fraction)
  alphas[:warmup_time] = np.linspace(start, end, warmup_time, dtype=np.float64)
  return alphas

def noise_schedule(schedule_name, start, end, n_timesteps, gammas=False):
    if schedule_name == 'quad':
        alphas = np.linspace(start ** 0.5, end ** 0.5, n_timesteps, dtype=np.float64) ** 2
    elif schedule_name == 'linear':
        alphas = np.linspace(start, end, n_timesteps, dtype=np.float64)
    elif schedule_name == 'warmup10':
        alphas = _warmup_alpha(start, end, n_timesteps, 0.1)
    elif schedule_name == 'warmup50':
        alphas = _warmup_alpha(start, end, n_timesteps, 0.5)
    elif schedule_name == 'const':
        alphas = end * np.ones(n_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(schedule_name)
    assert alphas.shape == (n_timesteps,)
    if gammas:
        return np.cumprod(alphas)
    else:
        return alphas

def generate_noisy_image(image, std_param):
    return np.sqrt(std_param) * image + np.sqrt(1 - std_param) * tf.random.normal(image.shape, stddev=1)