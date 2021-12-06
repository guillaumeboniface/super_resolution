# Creating the celebhq tfrecord dataset and uploading it to Google cloud storage
```
python3 -m sr3.scripts celebhq_to_gcs [celebhq image folder path] [google cloud project id]
```

# Running a training job
```
gcloud ai-platform jobs submit training $JOB_NAME \
    --staging-bucket=$STAGING_BUCKET \
    --job-dir=$JOB_DIR  \
    --package-path=$PACKAGE_PATH \
    --module-name=$MODULE_NAME \
    --region=$REGION \
    --config=config.yaml \
    -- \
    $TFR_BUCKET
```
To monitor the progress through tensorboard
```
tensorboard --logdir=$JOBDIR/tensorboard
```

# Train locally
```
python3 -m sr3.trainer.task $TFR_BUCKET $JOB_DIR --use_tpu=False
```

# Run training locally on a dummy model
```
python3 -m sr3.scripts dummy_train_run $TFR_BUCKET $JOB_DIR
``` 

# Running the notebooks
Running the notebooks requires installing locally the package
```
python3 -m build
pip3 install dist/sr3-0.1-py3-none-any.whl
```