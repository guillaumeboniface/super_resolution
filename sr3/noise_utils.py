import tensorflow as tf

def sample_noise_schedule(schedule: tf.Tensor, batch_size: int) -> tf.Tensor:
    n_timesteps = len(schedule)
    t = tf.experimental.np.random.randint(n_timesteps, size=batch_size)
    start_interval = tf.Tensor(schedule)[t]
    end_interval = tf.Tensor(schedule)[t + 1]
    noise = start_interval + tf.random.uniform() * (end_interval - start_interval)
    return noise

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