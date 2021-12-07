# Reproducing the SR3 paper
This repository focuses on partially reproducing the results of the [Image Super-Resolution via Iterate Refinement paper](https://iterative-refinement.github.io/). It focuses solely on the celebaHQ dataset and the 16x16 -> 128x128 task.

The repository is based on the model and methodology description from the paper. In some places though, details are lacking and guesses had to be made. What you can expect:
- The overall Unet architecture reproduces the implementation from [Denoising Diffusion Probabilistic Model](https://hojonathanho.github.io/diffusion/), something that was made easier from being able to access [their own paper implementation](https://github.com/hojonathanho/diffusion). It includes attention layers at lower resolutions and residual connections.
- The residual block composition is taken from [bigGAN p17](https://arxiv.org/pdf/1809.11096.pdf).
- The conditional instance normalization is taken from [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/pdf/1907.05600.pdf). One key difference is that we condition the normalization on Gamma which is a scalar (as opposed to a class, as in the paper conditional instance normalization comes from), obviating the need for an embedding. We used the scalar itself as a direct input to the conditional normalization layer.
- The default hyperparameters are based on the paper description: 256 batch size, 1M steps training, adam optimizer, 1e-4 learning rate with 50000 warmup steps, 0.2 dropout, 3 resnet blocks per unet layer, 128 channel dim, {1, 2, 4, 8, 8} depth multipliers. Despite all of this, the total number of parameters for our model is 480M, against 550M in the paper. Slight differences in the model architecture are likely accounting for the difference.
- The noise schedule is where the paper provides the least details, although it stresses its importance for the quality of the result. We implemented quadratic, linear, constant and warmup schedules. For training, we use a quadratic schedule, with a starting value for alpha of 1, an end value of 0.993 and 2000 timesteps.

# Training the model

## Relying on Google cloud storage
The scripts assume the data is stored in Google Cloud Storage (GCS) and you will want to create the Tensorflow Dataset from the celebhq_to_gcs script. In order to create the dataset in GCS and read it you will need to provide credentials to the script.
```
export GOOGLE_APPLICATION_CREDENTIALS=[filepath_to_credentials]
```

## Creating the celebhq tfrecord dataset and uploading it to Google cloud storage
Install python dependencies
```
pip3 install -r requirements.txt
```
Run the script
```
python3 -m sr3.scripts celebhq_to_gcs [celebhq image folder path] [google cloud project id]
```

## Running a training job on Google AI platform
In order to do this, you'll need to have a google cloud project created, as well as
(obviously) some kind of billing setup.
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

## Resuming a training job
You can resume a previous training job by passing the pass to the model saved thus
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
    --resume_model=$GS_RESUME_MODEL_PATH
```

## Tracking your training job performance
To monitor the progress through tensorboard
```
tensorboard --logdir=$JOB_DIR/tensorboard
```

## Train locally
This could be useful if you have a powerful machine. I don't.
```
python3 -m sr3.trainer.task $TFR_BUCKET $JOB_DIR --use_tpu=False
```