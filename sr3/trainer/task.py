import fire
import tensorflow as tf
from sr3.datasets.celebhq import *
from sr3.trainer.model import create_model
from sr3.utils import *

def train(bucket_name, job_dir, batch_size=32, use_tpu=True):
    if use_tpu:
        strategy_scope = initialize_tpu().scope()
        steps_per_execution = 50
    else:
        strategy_scope = DummyContextManager()
        steps_per_execution = None

    train_ds = get_dataset(bucket_name, batch_size, is_training=True)
    test_ds = get_dataset(bucket_name, batch_size, is_training=False)

    with strategy_scope:
        model = create_model(batch_size=batch_size)
        model.compile(optimizer='adam',
                steps_per_execution=steps_per_execution,
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=['mean_squared_error'])

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(job_dir, 'model', 'model_file'),
        save_weights_only=True,
        monitor='val_mean_squared_error',
        mode='max',
        save_best_only=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(job_dir, 'tensorboard'))
    
    steps_per_epoch = 4096 * 6 // batch_size
    validation_steps = (4096 + 1328) // batch_size
    
    model.fit(
        train_ds,
        epochs=5,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_ds,
        validation_steps=validation_steps,
        callbacks=[tensorboard_callback, model_checkpoint_callback])
    
    
if __name__ == '__main__':
  fire.Fire(train)