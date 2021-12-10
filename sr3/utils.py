from typing import Dict
from google.cloud import storage
from google.cloud.storage import bucket
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os
import tensorflow as tf
from tqdm import tqdm
from collections.abc import Iterable
import json

def warmup_adam_optimizer(learning_rate: float, warmup_steps: int = 10000) -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.Adam(
        learning_rate=WarmUpSchedule(learning_rate, warmup_steps),
    )

class WarmUpSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, target_learning_rate: float, warmup_steps: int, **kwargs) -> None:
        super(WarmUpSchedule, self).__init__(**kwargs)
        self.target_learning_rate = target_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step: int) -> tf.float32:
        warmup = tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32) * self.target_learning_rate
        return tf.math.minimum(self.target_learning_rate, warmup)

    def get_config(self) -> dict:
        config = {
            'target_learning_rate': self.target_learning_rate,
            'warmup_steps': self.warmup_steps
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def list_blobs(bucket_name: str) -> Iterable:
    """Lists all the blobs path in the bucket."""

    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name)

    return list(map(lambda x: 'gs://{}/{}'.format(bucket_name, x.name), blobs))

def get_tfr_files_path(folder_path: str) -> Iterable:
    return tf.io.gfile.glob(os.path.join(folder_path, '*.tfrec'))

def initialize_tpu() -> tf.distribute.Strategy:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
    strategy = tf.distribute.TPUStrategy(resolver)
    return strategy


def write_string_to_file(file_path: str, content: str) -> None:
    with tf.io.gfile.GFile(file_path, mode='w') as file:
        file.write(content)

def get_model_config(model_path: str) -> dict:
    parent_path = os.path.split(model_path)[0]
    parent_path = os.path.split(parent_path)[0]
    with tf.io.gfile.GFile(
            os.path.join(
                parent_path,
                'run_config.json'), mode='r') as config_file:
        config = json.loads(config_file.read())
        return config

def fix_resume_config(resumed_config: dict, current_config: dict) -> dict:
    for attribute in ['train_epochs', 'n_train_images', 'n_valid_images', 'resume_model']:
        resumed_config['previous_' + attribute] = resumed_config.get(attribute)
        resumed_config[attribute] = current_config.get(attribute)
    return resumed_config

class DummyContextManager(object):

  def __enter__(self):
    pass

  def __exit__(self, *args):
    pass