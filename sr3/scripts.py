import fire
import os
import tensorflow as tf
from sr3.datasets.celebhq import *
from sr3.utils import *
from tqdm import tqdm
from sr3.trainer.model import *
from sr3.trainer.task import train

def celebhq_to_gcs(src_path: str, dest_path, num_samples_per_record: int = 4096, full_res: int = 128, downscale_factor: int = 8) -> None:
    image_list = tf.io.gfile.glob(os.path.join(src_path, "*.jpg"))
    samples = list(map(lambda x: {"id": int(os.path.split(x.split(".")[0])[1]), "path": x}, image_list))

    num_samples = len(samples)
    num_tfrecords = num_samples // num_samples_per_record
    if num_samples % num_samples_per_record:
        num_tfrecords += 1  # add one record if there are any remaining samples

    dataset_metadata = {
        "name": "celebhq",
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
                    image = tf.io.decode_jpeg(tf.io.read_file(image_path))
                except Exception:
                    print("Couldn't package %s" % image_path)
                example = create_example(image, sample["id"], full_res, downscale_factor)
                writer.write(example.SerializeToString())
                 

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
    Convenience function to check whether the number of parameters match
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
    #tf.keras.utils.plot_model(model, "model_shape_info.png", show_shapes=True)

def dummy_train_run(tfr_folder, job_dir) -> None:
    train(tfr_folder, job_dir,
        train_epochs=3,
        use_tpu=False,
        channel_dim=16,
        channel_ramp_multiplier=(1, 2),
        num_resblock=1,
        n_train_images=128,
        n_valid_images=128,
        batch_size=16
        )

def downscale_img(src: str, dest: str, factor: int) -> None:
    image = tf.io.decode_jpeg(tf.io.read_file(src))
    h, w, c = image.shape
    image = tf.image.resize(image, (h // factor, w // factor))
    image = tf.cast(image, tf.uint8)
    tf.io.write_file(dest, tf.io.encode_jpeg(image))
    return
    
if __name__ == '__main__':
  fire.Fire()