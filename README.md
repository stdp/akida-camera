# Learning how to use Akida with a Camera Feed

This is an extremely basic way to utilise the Akida on-chip learning functionality. The demo will let you learn new classes of objects to recognise in the camera feed. This application is built to soley demonstrate how easy it is to use Akida's unique one-shot/few shot learning abilities. Instead of text labels, the system uses RGB LED's to represent the class that has been predicted.

### How Does Akida Learn?

In  native  learning  mode,  event  domain  neurons  learn quickly through a biological process known as Spike Time Dependent Plasticity (STDP), in which synapses that match an activation pattern are reinforced. BrainChip is utilizing a naturally homeostatic form of STDP learning in which neurons don’t saturate or switch off completely. 

STDP  is  possible  because  of  the  event-based processing method used by the Akida processor, and can be applied to incremental learning and one-shot or multi-shot learning.

**Read more:**

[What Is the Akida Event Domain Neural Processor?](https://brainchip.com/akida-foundations/)

[MetaTF Documentation](https://doc.brainchipinc.com) 

## Prerequisites

- Raspberry Pi 4 Compute Model with an IO Board
- PCI-e Akida Neuromorphic Processor [link](https://shop.brainchipinc.com/products/akida%E2%84%A2-development-kit-pcie-board)
- Raspberry Pi Camera Module
- WS2812 compatible RGB LEDs [link](https://core-electronics.com.au/neopixel-stick-8-x-ws2812-5050-rgb-led-with-integrated-drivers.html)
- Python 3.8 or higher

![Akida Neuromorphic SoC](https://i.imgur.com/g8YCnaX.jpeg)

![WS2812 compatible RGB LEDs](https://i.imgur.com/zg9xneM.png)

## Installation

### Setup the Hardware
1. Connect the Raspberry Pi Camera to the Raspberry Pi 4 Compute Model.
2. Ensure the Akida Neuromorphic Processor is correctly installed in the PCI-e slot on the IO Board and the drivers are installed. [link to instructions](https://brainchip.com/support-akida-pcie-board)
3. Conect the WS2812 compatible RGB LED's wiring to the 5v, GROUND and GPIO 18

### Prepare the Software Environment
1. Create a virtual environment with access to system packages (required for `picamera2` module):
   ```bash
   python3 -m venv venv --system-site-packages
   source venv/bin/activate
   ```

2. Clone the repository:
   ```bash
    git clone https://github.com/stdp/akida-camera.git
    cd akida-camera
    ```

3. Install the required Python modules:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To start the Inference system, ensure your virtual environment is activated and follow the steps below. Python must be run as sudo for the LEDs to function:

1. get your virtualenv python path
``` bash
which python

# example output "/home/neuro/projects/akida-camera/venv/bin/python"
```

2. copy the output of this and run the follow command, replacing python path with the output from the previous step:
```bash
sudo <python path> neurocam.py

# example command "sudo /home/neuro/projects/akida-camera/venv/bin/python akida_camera.py"
```

## Controlling the app

1. Press `1` to `0` on your keyboard to learn a new class, numbers 1 through 7 will be a RGB colour, 0 will be black (or off)

2. Press `s` to save the newly learnt classes into your model (delete the model file to re-initialise a blank slate)

## One-Shot learning as seen in the BrainChip Inc demo

Essentially this is a homemade version of this demonstration that BrainChip has built. You can view this in action here:

[![One-Shot Learning](http://img.youtube.com/vi/xeGAiWbKa7s/0.jpg)](https://youtu.be/xeGAiWbKa7s "One-Shot Learning")

[![Akida, how do you like them apples?](http://img.youtube.com/vi/p9pXN5-opGw/0.jpg)](https://www.youtube.com/watch?v=p9pXN5-opGw "Akida, how do you like them apples?")


View more One-shot / few shot learning demonstration videos: 
[User Community Platform](https://www.youtube.com/playlist?list=PLKZ8TPx-mIt2Mu3kXxm9BIW08lIDbvZdA)
