import fire
import tensorflow as tf
from sr3.datasets.celebhq import *
from sr3.utils import *
from tqdm import tqdm
from sr3.trainer.model import *

def celebhq_to_gcs(img_folder_path: str, project_id: str, num_samples_per_record: int = 4096) -> None:
    tfrecords_dir = "tfrecords"
    image_list = list(filter(lambda x: x[-4:] == '.jpg', os.listdir(img_folder_path)))
    samples = list(map(lambda x: {"id": int(x.split('.')[0]), "path": os.path.join(img_folder_path, x)}, image_list))

    num_samples = len(samples)
    num_tfrecords = num_samples // num_samples_per_record
    if num_samples % num_samples_per_record:
        num_tfrecords += 1  # add one record if there are any remaining samples

    if not os.path.exists(tfrecords_dir):
        os.makedirs(tfrecords_dir)  # creating TFRecords output folder

    for tfrec_num in tqdm(range(num_tfrecords)):
        record_samples = samples[tfrec_num * num_samples_per_record : min((tfrec_num+1) * num_samples_per_record, num_samples)]
        with tf.io.TFRecordWriter(
            tfrecords_dir + "/file_%.2i-%i.tfrec" % (tfrec_num, len(record_samples))
        ) as writer:
            for sample in record_samples:
                image_path = sample["path"]
                try:
                    image = tf.io.decode_jpeg(tf.io.read_file(image_path))
                except Exception:
                    print("Couldn't package %s" % image_path)
                example = create_example(image, sample["id"])
                writer.write(example.SerializeToString())

    bucket_name = create_bucket(project_id)
    upload_tfrecords_to_gcs(tfrecords_dir, bucket_name)

def model_wiring_test() -> None:
    """
    Convenience function to test locally that the graph builds
    """
    model = create_model(
        channel_dim=16,
        channel_ramp_multiplier=(1, 2),
        num_resblock=1
    )
    model.summary()

def model_size_check(use_deep_blocks: bool = False, resample_with_conv: bool = False) -> None:
    """
    Convenience function to check the number of parameters match
    the paper estimate (550M)
    """
    model = create_model(
        channel_dim=128,
        channel_ramp_multiplier=(1, 2, 4, 8, 8),
        num_resblock=3,
        batch_size=256,
        use_deep_blocks=use_deep_blocks,
        resample_with_conv=resample_with_conv
    )
    model.summary()
    
if __name__ == '__main__':
  fire.Fire()