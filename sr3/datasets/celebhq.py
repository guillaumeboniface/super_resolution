from typing import Callable
from tensorflow._api.v2 import data
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from sr3.utils import get_tfr_files_path
import tensorflow as tf
from collections.abc import Iterable
from sr3.noise_utils import *

def image_feature(value: str) -> tf.train.Feature:
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )


def int64_feature(value: int) -> tf.train.Feature:
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def downsample_image(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.resize(image, [16, 16], method='nearest')
    image = tf.image.resize(image,[128, 128], method='bicubic')
    return image


def create_example(image: tf.Tensor, id: int) -> tf.train.Example:
    feature = {
        "image": image_feature(image),
        "image_downsampled": image_feature(downsample_image(image)),
        "id": int64_feature(id),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord_fn(example: tf.train.Example) -> Iterable[tf.Tensor]:
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_downsampled": tf.io.FixedLenFeature([], tf.string),
        "id": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(example["image"], channels=3)
    image_downsampled = tf.io.decode_jpeg(example["image_downsampled"], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image_downsampled = tf.image.convert_image_dtype(image_downsampled, tf.float32)
    return image, image_downsampled


def augment_fn(images: Iterable[tf.Tensor]) -> tf.Tensor:
    flip = tf.experimental.numpy.random.randint(10)
    image, image_downsampled = images
    if flip % 2:
        image = tf.image.flip_left_right(image)
        image_downsampled = tf.image.flip_left_right(image)
    return tf.stack(image, image_downsampled)


def create_target_fn(noise_alpha_schedule: tf.Tensor) -> Callable:
    def target_fn(images: tf.Tensor) -> Iterable[Iterable]:
        batch_size = images.shape[0]
        image = images[:, 0]
        image_downsampled = images[:, 1]
        alpha_sample, gamma_sample, gamma_minus_one_sample = sample_noise_schedule(noise_alpha_schedule, batch_size)
        image_t = generate_noisy_image_batch(image, gamma_sample)
        image_t_minus_one = generate_noisy_image_minus_one_batch(image, image_t, gamma_sample, gamma_minus_one_sample, alpha_sample)
        return [[image_downsampled, image_t, gamma_sample], image_t_minus_one]
    return target_fn


def get_dataset(bucket_name: str, batch_size: int, noise_alpha_schedule: tf.Tensor, is_training: bool = True) -> tf.data.Dataset:
    if is_training:
        tfr_files_path = get_tfr_files_path(bucket_name)[:-2]
    else:
        tfr_files_path = get_tfr_files_path(bucket_name)[-2:]
    raw_dataset = tf.data.TFRecordDataset(tfr_files_path)

    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed

    dataset = raw_dataset.map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
    dataset = dataset.with_options(
        ignore_order
    )
    if is_training:
        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat()
        dataset = dataset.map(augment_fn, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.map(create_target_fn(noise_alpha_schedule), num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset