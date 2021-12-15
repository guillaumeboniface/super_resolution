import fire
import os
import tensorflow as tf
from sr3.trainer.model import create_model
from sr3.trainer.task import train


def model_wiring_test() -> None:
    """
    Convenience function to test locally that the graph builds
    """
    model = create_model(
        img_shape=(128,128,3),
        channel_dim=16,
        channel_ramp_multiplier=(1, 2),
        batch_size=32,
        num_resblock=1,
    )
    model.summary()

def model_size_check(use_deep_blocks: bool = False, resample_with_conv: bool = False) -> None:
    """
    Convenience function to check whether the number of parameters match
    the paper estimate (550M)
    """
    model = create_model(
        img_shape=(128,128,3),
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