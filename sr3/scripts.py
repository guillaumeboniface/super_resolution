import fire
import tensorflow as tf
from datasets.celebhq import *
from utils import *
from tqdm import tqdm

def celebhq_to_gcs(img_folder_path, project_id, num_samples_per_record=4096):
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
    
if __name__ == '__main__':
  fire.Fire()