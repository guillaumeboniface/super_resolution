from typing import Dict
from google.cloud import storage
from google.cloud.storage import bucket
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os
import tensorflow as tf
from tqdm import tqdm
from collections.abc import Iterable

def warmup_adam_optimizer(learning_rate: float, warmup_steps: int = 10000) -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.Adam(
        learning_rate=WarmUpSchedule(learning_rate, warmup_steps),
    )

class WarmUpSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, target_learning_rate: float, warmup_steps: int) -> None:
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

def list_blobs(bucket_name: str) -> Iterable[str]:
    """Lists all the blobs path in the bucket."""

    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name)

    return list(map(lambda x: 'gs://{}/{}'.format(bucket_name, x.name), blobs))

def get_tfr_files_path(bucket_name: str) -> Iterable[str]:
    return list(filter(lambda x: x[-5:] == "tfrec", list_blobs(bucket_name)))

def initialize_tpu() -> tf.distribute.Strategy:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
    strategy = tf.distribute.TPUStrategy(resolver)
    return strategy

def create_bucket(project_id: str, bucket_prefix: str = 'celebahq128') -> str:
    gcs_service = build('storage', 'v1')

    # Generate a random bucket name to which we'll upload the file.
    import uuid
    bucket_name = bucket_prefix + str(uuid.uuid1())

    body = {
        'name': bucket_name,
        # For a full list of locations, see:
        # https://cloud.google.com/storage/docs/bucket-locations
        'location': 'eu',
    }
    gcs_service.buckets().insert(project=project_id, body=body).execute()
    print('Created bucket named {}'.format(bucket_name))
    return bucket_name

def upload_tfrecords_to_gcs(source_dir: str, bucket_name: str) -> None:
    for file_name in tqdm(filter(lambda x: x[-5:] == 'tfrec', os.listdir(source_dir))):
        media = MediaFileUpload(os.path.join(source_dir, file_name), 
                                mimetype='text/plain',
                                resumable=True)

        gcs_service = build('storage', 'v1')
        request = gcs_service.objects().insert(bucket=bucket_name, 
                                            name=os.path.join('tfrecords', file_name),
                                            media_body=media)

        response = None
        while response is None:
            # _ is a placeholder for a progress object that we ignore.
            # (Our file is small, so we skip reporting progress.)
            _, response = request.next_chunk()

    print('Upload complete')

def write_string_to_gcs(file_path: str, content: str) -> None:
    with tf.io.gfile.GFile(file_path, mode='w') as file:
        file.write(content)

class DummyContextManager(object):

  def __enter__(self):
    pass

  def __exit__(self, *args):
    pass