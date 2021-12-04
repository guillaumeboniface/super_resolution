from tensorflow._api.v2 import data
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from sr3.utils import get_tfr_files_path
import tensorflow as tf
import numpy as np
from collections.abc import Iterable

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

def augment_fn(images: Iterable[tf.Tensor]) -> Iterable[tf.Tensor]:
    flip = np.random.randint(10)
    image, image_downsampled = images
    if flip % 2:
        image = tf.image.flip_left_right(image)
        image_downsampled = tf.image.flip_left_right(image)
    return image, image_downsampled

def get_dataset(bucket_name: str, batch_size: int, is_training: bool = True) -> tf.data.Dataset:
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
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset