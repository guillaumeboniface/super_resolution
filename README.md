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

# Train locally
```
python3 -m sr3.trainer.task $BUCKET_NAME $JOB_DIR --use_tpu=False
```