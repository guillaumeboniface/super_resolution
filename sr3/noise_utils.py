import tensorflow as tf
from collections.abc import Iterable

def sample_noise_schedule(alpha_schedule: tf.Tensor, batch_size: int) -> Iterable[tf.Tensor]:
    n_timesteps = len(alpha_schedule)
    t = tf.experimental.numpy.random.randint(1, high=n_timesteps, size=batch_size)
    random_interval = tf.random.uniform((batch_size,), dtype=tf.float64)

    start_interval = tf.gather(alpha_schedule, t)
    end_interval = tf.gather(alpha_schedule, t - 1)
    alpha = start_interval + random_interval * (end_interval - start_interval)

    gamma_schedule = tf.math.cumprod(alpha_schedule)
    start_interval = tf.gather(gamma_schedule, t)
    end_interval = tf.gather(gamma_schedule, t - 1)
    gamma = start_interval + random_interval * (end_interval - start_interval)

    start_interval = tf.gather(gamma_schedule, t - 1)
    end_interval = tf.gather(gamma_schedule, tf.math.maximum(t - 2, tf.zeros((batch_size,), dtype=tf.int64)))
    gamma_minus_one = start_interval + random_interval * (end_interval - start_interval)

    return tf.cast(alpha, tf.float32), tf.cast(gamma, tf.float32), tf.cast(gamma_minus_one, tf.float32)

def _warmup_alpha(start: float, end: float, n_timesteps: int, warmup_fraction: float) -> tf.Tensor:
    alphas = end * tf.ones(n_timesteps, dtype=tf.float64)
    warmup_time = int(n_timesteps * warmup_fraction)
    alphas[:warmup_time] = tf.linspace(start, end, warmup_time)
    return alphas

def noise_schedule(schedule_name: str, start: float, end: float, n_timesteps: int, gammas: bool = False) -> tf.Tensor:
    start = tf.cast(start, tf.float64)
    end = tf.cast(end, tf.float64)
    if schedule_name == 'quad':
        alphas = tf.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule_name == 'linear':
        alphas = tf.linspace(start, end, n_timesteps)
    elif schedule_name == 'warmup10':
        alphas = _warmup_alpha(start, end, n_timesteps, 0.1)
    elif schedule_name == 'warmup50':
        alphas = _warmup_alpha(start, end, n_timesteps, 0.5)
    elif schedule_name == 'const':
        alphas = end * tf.ones(n_timesteps, dtype=tf.float64)
    else:
        raise NotImplementedError(schedule_name)
    assert alphas.shape == (n_timesteps,)
    if gammas:
        return tf.math.cumprod(alphas)
    else:
        return alphas

def generate_noisy_image(image: tf.Tensor, std_param: tf.float32) -> tf.Tensor:
    return tf.math.sqrt(std_param) * image + tf.math.sqrt(1 - std_param) * tf.random.normal(image.shape, stddev=1.)

def generate_noisy_image_batch(image: tf.Tensor, std_param: tf.float32) -> tf.Tensor:
    std_param = tf.reshape(std_param, [std_param.shape[0], 1, 1, 1])
    return tf.math.sqrt(std_param) * image + tf.math.sqrt(1 - std_param) * tf.random.normal(image.shape, stddev=1.)

def generate_noisy_image_minus_one_batch(
    image: tf.Tensor,
    noisy_img: tf.Tensor,
    gamma: tf.float32,
    gamma_minus_one: tf.float32,
    alpha: tf.float32) -> tf.Tensor:
    """
    Calculating image at t - 1 knowing the noise distribution,
    the image at t_zero and the image at t. For the original formula
    see paper page 3: https://arxiv.org/pdf/2104.07636.pdf
    """
    gamma = tf.reshape(gamma, [gamma.shape[0], 1, 1, 1])
    alpha = tf.reshape(alpha, [alpha.shape[0], 1, 1, 1])
    gamma_minus_one = tf.reshape(gamma_minus_one, [gamma_minus_one.shape[0], 1, 1, 1])
    mu = tf.math.sqrt(gamma_minus_one) * (1 - alpha) / (1 - gamma) * image + \
            tf.math.sqrt(alpha) * (1 - gamma_minus_one) / (1 - gamma) * noisy_img
    sigma = tf.math.sqrt((1 - gamma_minus_one) * (1 - alpha) / (1 - gamma))
    return mu + sigma * tf.random.normal(image.shape, stddev=1.)