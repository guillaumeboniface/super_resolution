import fire
import os
import tensorflow as tf
import json
from collections.abc import Iterable
from sr3.dataset import get_dataset, IMAGE_SHAPE
from sr3.trainer.model import create_model, custom_objects
import sr3.utils as utils
from sr3.noise_utils import noise_schedule

def train(
    tfr_folder: str,
    job_dir: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_epochs: int = 420,
    epoch_factor: int = 1,
    learning_rate: float = 1e-4,
    learning_warmup_steps: int = 10000,
    dropout: float = 0.2,
    noise_schedule_shape: str = 'linear',
    noise_schedule_start: float = 0.995,
    noise_schedule_end: float = 0.998,
    noise_schedule_steps: int = 2000,
    channel_dim: int = 128,
    channel_ramp_multiplier: Iterable = (1, 2, 4, 8, 8),
    attention_resolution: Iterable = (8,),
    num_resblock: int = 3,
    use_deep_blocks: bool = False,
    resample_with_conv: bool = True,
    use_tpu: bool = True,
    tpu_steps_per_execution: int = None,
    resume_model: str = None) -> None:

    saved_args = locals()

    if use_tpu:
        strategy_scope = utils.initialize_tpu().scope()
        steps_per_execution = tpu_steps_per_execution
    else:
        strategy_scope = utils.DummyContextManager()
        steps_per_execution = None

    noise_alpha_schedule = noise_schedule(noise_schedule_shape, noise_schedule_start, noise_schedule_end, noise_schedule_steps)
    train_ds, n_train_images = get_dataset(tfr_folder, batch_size, noise_alpha_schedule, valid_ratio, is_training=True)
    test_ds, n_valid_images = get_dataset(tfr_folder, batch_size, noise_alpha_schedule, valid_ratio, is_training=False)

    n_train_images *= epoch_factor #lengthen each "epoch" by going over the dataset epoch_factor time

    with strategy_scope:
        if resume_model:
            model = tf.keras.models.load_model(resume_model, custom_objects=custom_objects)
            config = utils.get_model_config(resume_model)
            config = utils.fix_resume_config(config, saved_args)
        else:
            model = create_model(
                batch_size=batch_size,
                img_shape=IMAGE_SHAPE,
                dropout=dropout,
                channel_dim=channel_dim,
                channel_ramp_multiplier=channel_ramp_multiplier,
                attention_resolution=attention_resolution,
                num_resblock=num_resblock,
                use_deep_blocks=use_deep_blocks,
                resample_with_conv=resample_with_conv)
            model.compile(optimizer=utils.warmup_adam_optimizer(learning_rate, learning_warmup_steps),
                steps_per_execution=steps_per_execution,
                loss=tf.keras.losses.MeanSquaredError())
            config = saved_args
        utils.write_string_to_file(os.path.join(job_dir, 'run_config.json'), json.dumps(config, indent=4))
        

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(job_dir, 'models', 'model.{epoch:04d}-{val_loss:.2f}'),
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(job_dir, 'tensorboard'),
        update_freq=120,
        write_steps_per_second=True,
        histogram_freq=120)
    
    steps_per_epoch = n_train_images // batch_size
    validation_steps = n_valid_images // batch_size
    
    model.fit(
        train_ds,
        epochs=train_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_ds,
        validation_steps=validation_steps,
        callbacks=[tensorboard_callback, model_checkpoint_callback])

    model.save(os.path.join(job_dir, 'models', 'model_final'))
    
    
if __name__ == '__main__':
  fire.Fire(train)