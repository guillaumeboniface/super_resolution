# Reproducing the Image Super-Resolution via Iterative Refinement paper
This repository focuses on partially reproducing the results of the [Image Super-Resolution via Iterate Refinement paper](https://iterative-refinement.github.io/). It focuses solely on the celebaHQ dataset and the 16x16 -> 128x128 task.

The repository is based on the model and methodology description from the paper. In some places though, details are lacking and guesses had to be made. What you can expect:
- The overall Unet architecture reproduces the implementation from [Denoising Diffusion Probabilistic Model](https://hojonathanho.github.io/diffusion/), something that was made easier from being able to access [their own paper implementation](https://github.com/hojonathanho/diffusion). It includes attention layers at lower resolutions and residual connections.
- The residual block composition is taken from [bigGAN p17](https://arxiv.org/pdf/1809.11096.pdf).
- The conditional instance normalization is taken from [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/pdf/1907.05600.pdf).
- The default hyperparameters are based on the paper description: 256 batch size, 1M steps training, adam optimizer, 1e-4 learning rate with 50000 warmup steps, 0.2 dropout, 3 resnet blocks per unet layer, 128 channel dim, {1, 2, 4, 8, 8} depth multipliers. Despite all of this, the total number of parameters for our model is 480M, against 550M in the paper. Slight differences in the model architecture are likely accounting for the difference.
- The noise schedule is where the paper provides the least details, although it stresses its importance for the quality of the result. We implemented quadratic, linear, constant and warmup schedules. For training, we use a quadratic schedule, with a starting value for alpha of 1, an end value of 0.993 and 2000 timesteps.

# Training the model

## Creating the tfrecord dataset and uploading it to Google cloud storage
Install python dependencies
```
pip3 install -r requirements.txt
```
Run the script. You can point to Google Cloud Storage should you want to (required to run the training job on Google Cloud platform)
```
python3 -m sr3.dataset [image folder path] [tfrec destination path] --file_format=[png or jpg]
```

## Running a training job on Google AI platform
In order to do this, you'll need to have a google cloud project created, as well as some kind of billing setup. You'll need also to [install the gcloud SDK](https://cloud.google.com/sdk/docs/install) and configure it to your project. For more info on the parameters for this command, refer to the [google cloud documentation](https://cloud.google.com/ai-platform/training/docs/training-jobs). All the path provided should be google cloud storage path. JOB_DIR is where the model and tensorboard data will be stored, TFR_PATH the path to the folder with the .tfrec files.
```
gcloud ai-platform jobs submit training $JOB_NAME \
    --staging-bucket=$STAGING_BUCKET \
    --job-dir=$JOB_DIR  \
    --package-path=./sr3 \
    --module-name=sr3.trainer.task \
    --region=$REGION \
    --config=config.yaml \
    -- \
    $TFR_PATH
```

## Resuming a training job
You can resume a previous training job by passing the path to the model folder
```
gcloud ai-platform jobs submit training $JOB_NAME \
    --staging-bucket=$STAGING_BUCKET \
    --job-dir=$JOB_DIR  \
    --package-path=./sr3 \
    --module-name=sr3.trainer.task \
    --region=$REGION \
    --config=config.yaml \
    -- \
    $TFR_PATH
    --resume_model=$GS_RESUME_MODEL_PATH
```

## Tracking your training job performance
To monitor the progress of a training run through tensorboard
```
tensorboard --logdir=$JOB_DIR/tensorboard
```

## Train locally
If you want to train locally, you can do so. JOB_DIR is where the model and tensorboard data will be stored, TFR_PATH the path to the folder with the .tfrec files.
```
python3 -m sr3.trainer.task $TFR_PATH $JOB_DIR --use_tpu=False
```

## Inference
You can perform inference from the command line.
```
python3 -m sr3.evaluation.infer $MODEL_PATH $IMG_SRC $IMG_DEST
```