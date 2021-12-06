import fire
import os
import tensorflow as tf
import json
from sr3.datasets.celebhq import *
from sr3.trainer.model import create_model
from sr3.utils import *
from sr3.noise_utils import *

def train(
    bucket_name: str,
    job_dir: str,
    batch_size: int = 256,
    n_train_images: int = 4096 * 6,
    n_valid_images: int = 4096 + 1328,
    train_epochs: int = 10000,
    learning_rate: float = 1e-4,
    learning_warmup_steps: int = 10000,
    dropout: float = 0.2,
    noise_schedule_shape: str = 'quad',
    noise_schedule_start: float = 1.,
    noise_schedule_end: float = 0.993,
    noise_schedule_steps: int = 2000,
    channel_dim: int = 128,
    channel_ramp_multiplier: Iterable[int] = (1, 2, 4, 8, 8),
    attention_resolution: Iterable[int] = (8,),
    num_resblock: int = 3,
    use_deep_blocks: bool = False,
    resample_with_conv: bool = False,
    use_tpu: bool = True) -> None:

    saved_args = locals()

    if use_tpu:
        strategy_scope = initialize_tpu().scope()
        steps_per_execution = 50
    else:
        strategy_scope = DummyContextManager()
        steps_per_execution = None

    noise_alpha_schedule = noise_schedule(noise_schedule_shape, noise_schedule_start, noise_schedule_end, noise_schedule_steps)
    train_ds = get_dataset(bucket_name, batch_size, noise_alpha_schedule, is_training=True)
    test_ds = get_dataset(bucket_name, batch_size, noise_alpha_schedule, is_training=False)

    write_string_to_gcs(os.path.join(job_dir, 'run_config.json'), json.dumps(saved_args, indent=4))

    with strategy_scope:
        model = create_model(
            batch_size=batch_size,
            dropout=dropout,
            channel_dim=channel_dim,
            channel_ramp_multiplier=channel_ramp_multiplier,
            attention_resolution=attention_resolution,
            num_resblock=num_resblock,
            use_deep_blocks=use_deep_blocks,
            resample_with_conv=resample_with_conv)
        model.compile(optimizer=warmup_adam_optimizer(learning_rate, learning_warmup_steps),
                steps_per_execution=steps_per_execution,
                loss=tf.keras.losses.MeanSquaredError())

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(job_dir, 'models', 'model.{epoch:04d}-{val_loss:.2f}'),
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(job_dir, 'tensorboard'),
        update_freq='epoch',
        write_steps_per_second=True,
        histogram_freq=100)
    
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