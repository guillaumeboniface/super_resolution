# utility to populate env variables before running the train command line
# replace the brackets by value then run the script to be ready to run the
# training command
now=$(date +"%Y%m%d_%H%M%S")
export JOB_NAME="sr3_train_$now"
export STAGING_BUCKET=[your staging bucket]
export TFR_PATH=[folder path with tfds and train run data, including models]
export JOB_DIR=[job folder]/train_output_$now
export PACKAGE_PATH=./sr3
export MODULE_NAME=sr3.trainer.task
export REGION=[region to run]