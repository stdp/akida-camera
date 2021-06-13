import time
import os
import cv2
import threading
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow_datasets as tfds
import numpy as np
from pynput import keyboard

from akida_models import mobilenet_edge_imagenet_pretrained
from cnn2snn import convert
from akida import Model, FullyConnected


MODEL_FBZ = "models/edge_learning_example.fbz"

CAMERA_SRC = 0
INFERENCE_PER_SECOND = 2

TEXT_COLOUR = (190, 30, 255)

NUM_NEURONS_PER_CLASS = 1
NUM_WEIGHTS = 350
NUM_CLASSES = 10

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

TARGET_WIDTH = 224
TARGET_HEIGHT = 224

NEURON_KEYS = [str(i) for i in range(10)]
SAVE_BUTTON = 's'


"""
Method to initialise an Akida model if one doesn't exist
"""


def initialise():
    ds, ds_info = tfds.load("coil100:2.*.*", split="train", with_info=True)
    model_keras = mobilenet_edge_imagenet_pretrained()
    # convert it to akida
    model_ak = convert(model_keras, input_scaling=(128, 128))

    # remove the last layer of network, replace with Akida learning layer
    model_ak.pop_layer()
    layer_fc = FullyConnected(
        name="akida_edge_layer",
        num_neurons=NUM_CLASSES * NUM_NEURONS_PER_CLASS,
        activations_enabled=False,
    )
    # add learning layer to end of model
    model_ak.add(layer_fc)
    model_ak.compile(
        num_weights=NUM_WEIGHTS, num_classes=NUM_CLASSES, learning_competition=0.1
    )
    # save new model
    model_ak.save(MODEL_FBZ)


# create a model if one doesnt exist
if not os.path.exists(MODEL_FBZ):
    print("Initialising Akida model")
    initialise()


"""
Class to capture key presses to save/learn
"""


class Controls:

    def __init__(self, inference):
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()
        self.inference = inference

    def on_press(self, key):
        global inference
        try:

            if key.char in NEURON_KEYS:
                print("learned class {}".format(int(key.char)))
                self.inference.learn(int(key.char))

            if key.char == SAVE_BUTTON:
                print("saved model to {}".format(MODEL_FBZ))
                self.inference.save()

        except AttributeError:
            pass

    def on_release(self, key):
        if key == keyboard.Key.esc:
            return False


"""
Class to capture video feed from webcam
"""


class Camera:
    def __init__(self):
        self.stream = VideoStream(
            src=CAMERA_SRC, resolution=(FRAME_WIDTH, FRAME_HEIGHT)
        ).start()
        self.label = 0

    def get_frame(self):
        frame = cv2.resize(self.stream.read(), (TARGET_WIDTH, TARGET_HEIGHT))
        return frame

    def get_input_array(self):
        frame = cv2.resize(self.stream.read(), (TARGET_WIDTH, TARGET_HEIGHT))
        input_array = img_to_array(frame)
        input_array = np.array([input_array], dtype="uint8")
        return input_array

    def show_frame(self):
        frame = self.label_frame(self.stream.read())
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF

    def label_frame(self, frame):
        frame = cv2.putText(
            frame,
            str(self.label),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            TEXT_COLOUR,
            5,
            cv2.LINE_AA,
        )
        return frame

    def set_label(self, label):
        self.label = label


"""
Class to run inference over frames from the webcam
"""


class Inference:
    def __init__(self, camera):
        # init the camera
        self.camera = camera
        # run inference in separate thread
        self.t1 = threading.Thread(target=self.infer)
        self.t1.start()
        # load the akida model
        self.model_ak = Model(filename=MODEL_FBZ)

    def infer(self):
        while True:
            input_array = camera.get_input_array()
            predictions = self.model_ak.predict(input_array, num_classes=NUM_CLASSES)
            self.camera.set_label(predictions[0])
            time.sleep(1 / INFERENCE_PER_SECOND)

    def learn(self, neuron):
        input_array = self.camera.get_input_array()
        self.model_ak.fit(input_array, neuron)

    def save(self):
        self.model_ak.save(MODEL_FBZ)


camera = Camera()
inference = Inference(camera)
controls = Controls(inference)

# main loop to display camera feed
while True:
    camera.show_frame()
