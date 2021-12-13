from typing import Callable
import os
from tensorflow._api.v2 import data
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from tensorflow.python.training.input import batch
from sr3.utils import *
import tensorflow as tf
from collections.abc import Iterable
from sr3.noise_utils import *
import fire

IMAGE_SHAPE = (128, 128, 3)

def image_feature(value: str) -> tf.train.Feature:
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )


def int64_feature(value: int) -> tf.train.Feature:
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def downsample_image(image: tf.Tensor, full_res: int, downscale_factor: int) -> tf.Tensor:
    image = tf.image.resize(image, [full_res // downscale_factor, full_res // downscale_factor], method='nearest')
    image = tf.image.resize(image,[full_res, full_res], method='bicubic')
    return tf.cast(image, tf.uint8)


def create_example(image: tf.Tensor, id: int, full_res: int, downscale_factor: int) -> tf.train.Example:
    feature = {
        "image": image_feature(image),
        "image_downsampled": image_feature(downsample_image(image, full_res, downscale_factor)),
        "id": int64_feature(id),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord_fn(example: tf.train.Example) -> tf.Tensor:
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
    return tf.stack([image, image_downsampled])


def create_target_fn(noise_alpha_schedule: tf.Tensor, batch_size: int) -> Callable:
    def target_fn(images: tf.Tensor) -> Iterable:
        image = tf.reshape(images[:, 0], (batch_size,) + IMAGE_SHAPE) # reshape to provide tensor shape at graph build time
        image_downsampled = tf.reshape(images[:, 1], (batch_size,) + IMAGE_SHAPE)
        alpha_sample, gamma_sample = sample_noise_schedule(noise_alpha_schedule, batch_size)
        image_t, noise = generate_noisy_image_batch(image, gamma_sample)
        # gamma should be of shape (batch, 1), this is a hack to circumvent the fact that a data pipeline cannot pass
        # a ragged shape
        gamma_sample = tf.broadcast_to(tf.reshape(gamma_sample, (batch_size, 1, 1, 1)), (batch_size,) + IMAGE_SHAPE)
        return (tf.stack([image_downsampled, image_t, gamma_sample], axis=1), noise)
    return target_fn


def get_dataset(tfr_folder: str, batch_size: int, noise_alpha_schedule: tf.Tensor, valid_ratio: float, is_training: bool = True) -> Iterable:
    with tf.io.gfile.GFile(os.path.join(tfr_folder, "metadata.json")) as metadata_file:
        metadata = json.loads(metadata_file)

    num_training_files = metadata.get("num_samples") * (1 - valid_ratio) // metadata.get("samples_per_file")
    
    if is_training:
        num_samples = num_training_files * metadata.get("samples_per_file")
        tfr_files_path = get_tfr_files_path(tfr_folder)[:num_training_files]
    else:
        num_samples = metadata.get("num_samples") - num_training_files * metadata.get("samples_per_file")
        tfr_files_path = get_tfr_files_path(tfr_folder)[num_training_files:]
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
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(create_target_fn(noise_alpha_schedule, batch_size), num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, num_samples


def dataset_to_gcs(
        src_path: str,
        dest_path,
        file_format: str = "jpg",
        num_samples_per_record: int = 4096,
        full_res: int = 128,
        downscale_factor: int = 8) -> None:
    image_list = tf.io.gfile.glob(os.path.join(src_path, "*." + file_format))
    samples = list(map(lambda x: {"id": int(os.path.split(x.split(".")[0])[1]), "path": x}, image_list))

    if file_format == "jpg":
        decode = tf.io.decode_jpeg
    elif file_format == "png":
        decode = tf.io.decode_png
    else:
        raise NotImplementedError("Format not supported")

    num_samples = len(samples)
    num_tfrecords = num_samples // num_samples_per_record
    if num_samples % num_samples_per_record:
        num_tfrecords += 1  # add one record if there are any remaining samples

    dataset_metadata = {
        "num_files": num_tfrecords,
        "samples_per_file": num_samples_per_record,
        "num_samples": num_samples,
        "full_res": full_res,
        "downscale_factor": downscale_factor
    }
    write_string_to_file(os.path.join(dest_path, "metadata.json"), json.dumps(dataset_metadata, indent=4))
    for tfrec_num in tqdm(range(num_tfrecords)):
        record_samples = samples[tfrec_num * num_samples_per_record : min((tfrec_num+1) * num_samples_per_record, num_samples)]
        with tf.io.TFRecordWriter(
            dest_path + "/file_%.2i-%i.tfrec" % (tfrec_num, len(record_samples))
        ) as writer:
            for sample in record_samples:
                image_path = sample["path"]
                try:
                    image = decode(tf.io.read_file(image_path))
                except Exception:
                    print("Couldn't package %s" % image_path)
                example = create_example(image, sample["id"], full_res, downscale_factor)
                writer.write(example.SerializeToString())

if __name__ == '__main__':
  fire.Fire(dataset_to_gcs)