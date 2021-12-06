# Install python dependencies to run on a local machine
```
pip3 install -r requirements.txt
```

# Relying on Google cloud storage
The scripts assume the data is stored in Google Cloud Storage (GCS) and you will want to create the
Tensorflow Dataset from the celebhq_to_gcs script. In order to create the dataset in GCS and read it
you will need to provide credentials to the script
```
export GOOGLE_APPLICATION_CREDENTIALS=[filepath_to_credentials]
```

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
tensorboard --logdir=$JOB_DIR/tensorboard
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
If you're using an environment, you'll want to make it available in your notebook
```
python3 -m ipykernel install --user --name=super_resolution
```