# Running the notebooks
I used notebooks to visualise different noise schedules, as well
as getting a feel for the function generating noise for training.
Running the notebooks requires installing locally the package.
```
python3 -m build
pip3 install dist/sr3-0.1-py3-none-any.whl
```
If you're using an environment, you'll want to make it available in your notebook
```
python3 -m ipykernel install --user --name=super_resolution
```