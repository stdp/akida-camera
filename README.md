# Learning how to use Akida with a Camera Feed

This is an extremely basic way to utilise the Akida on-chip learning functionality. The demo will let you learn new classes of objects to recognise in the camera feed. This application is built to soley demonstrate how easy it is to use Akida's unique one-shot/few shot learning abilities.


## How Does Akida Learn?


In  native  learning  mode,  event  domain  neurons  learn quickly through a biological process known as Spike Time Dependent Plasticity (STDP), in which synapses that match an activation pattern are reinforced. BrainChip is utilizing a naturally homeostatic form of STDP learning in which neurons don’t saturate or switch off completely. 

STDP  is  possible  because  of  the  event-based processing method used by the Akida processor, and can be applied to incremental learning and one-shot or multi-shot learning. 

**Read more:**

[What Is the Akida Event Domain Neural Processor?](https://brainchipinc.com/what-is-the-akida-event-domain-neural-processor/)

[MetaTF Documentation](https://doc.brainchipinc.com)


### Setting up the Akida development environment

1. Go to `https://www.anaconda.com/download/` and download the installer
2. Install Anaconda by running `bash Anaconda-latest-Linux-x86_64.sh`
3. Once installed, create a conda environment `conda create --name akida_env python=3.7`
4. Activate the new conda environment `conda activate akida_env`
5. Install the python dependencies `pip install -r requirements.txt`


### Running and using the app

1. run `python3 akida_camera.py`
2. Press `1` to `0` on your keyboard to learn a new class
3. Press `s` to save the newly learnt classes into your model


### Customise the labels

1. Edit the dict `LABELS = {0: "Background"}`
2. Add as many labels as required, ensure there are enough classes available in the model `NUM_CLASSES = 10`


### Record a demonstration

1. Change the output setting `OUTPUT = True`
2. Enter a filename if required, defaults to `out.avi`
> **Important:** Remember to contact `sales@brainchipinc.com` to seek permission before publishing any demonstration videos


## One-Shot learning as seen in the BrainChip Inc demo

Essentially this is a homemade version of this demonstration that BrainChip has built. You can view this in action here:

[![One-Shot Learning](http://img.youtube.com/vi/xeGAiWbKa7s/0.jpg)](https://youtu.be/xeGAiWbKa7s "One-Shot Learning")

[![Akida, how do you like them apples?](http://img.youtube.com/vi/p9pXN5-opGw/0.jpg)](https://www.youtube.com/watch?v=p9pXN5-opGw "Akida, how do you like them apples?")


View more One-shot / few shot learning demonstration videos: 
[User Community Platform](https://www.youtube.com/playlist?list=PLKZ8TPx-mIt2Mu3kXxm9BIW08lIDbvZdA)
