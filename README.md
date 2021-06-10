# Learning how to use the Akida with a Camera Feed

This is an extremely basic way to utilise the Akida on-chip learning functionality. The demo will let you learn new classes of objects to recognise in the camera feed.

## Setting up the Akida development evironment

1. Go to `https://www.anaconda.com/download/` and download intaller
2. Install anaconda `bash Anaconda-latest-Linux-x86_64.sh`
3. Create conda environment `conda create --name akida_env python=3.6`
4. Activate conda environement `conda activate akida_env`
5. Install python dependencies `pip install -r requirements.txt`

## Running and using the example

1. `python3 akida_camera.py`
2. Press `1`, `2` or `3` to save learn a new class
3. Press `s` to save the newly learnt features into your model


### Read More

Read all the documentation at https://doc.brainchipinc.com