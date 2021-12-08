# Run the unit tests
I wrote a few tests to double-check the noise functions behaved as intended.
You can run them thus. 
```
python3 -m unittest
```

# Run training locally on a dummy model
This runs an extremely small version of the model on small batches.
This was useful at development time to debug.
```
python3 -m sr3.scripts dummy_train_run $TFR_PATH $JOB_DIR
```