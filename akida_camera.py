import time
import os
import cv2
import threading
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow_datasets as tfds
import numpy as np

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

TARGET_WIDTH = 224
TARGET_HEIGHT = 224

NEURON_KEYS = {
    48: 0,
    49: 1,
    50: 2,
    51: 3,
    52: 4,
    53: 5,
    54: 6,
    55: 7,
    56: 8,
    57: 9,
}

SAVE_BUTTON = 115


"""
Method to initialise an Akida model if one doesn't exist
"""


def initialise(self):
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


"""
Class to capture key presses to save/learn
"""


class Controls:
    def __init__(self, camera):
        self.camera = camera

    def capture(self):
        n = cv2.waitKey(33)
        if n in NEURON_KEYS:
            print("learned class {}".format(NEURON_KEYS[n]))
            self.learn(NEURON_KEYS[n])

        if n == SAVE_BUTTON:
            print("saved model")
            self.save()
        self.camera.show_frame()

    def learn(self, neuron):
        global model_ak
        input_array = self.camera.get_input_array()
        model_ak.fit(input_array, neuron)

    def save(self):
        global model_ak
        model_ak.save(MODEL_FBZ)


"""
Class to capture video feed from webcam
"""


class Camera:
    def __init__(self):
        self.stream = VideoStream(
            src=CAMERA_SRC, resolution=(TARGET_WIDTH, TARGET_HEIGHT)
        ).start()

    def get_frame(self):
        frame = self.stream.read()
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        return frame

    def get_input_array(self):
        frame = cv2.resize(self.stream.read(), (TARGET_WIDTH, TARGET_HEIGHT))
        input_array = img_to_array(frame)
        input_array = np.array([input_array], dtype="uint8")
        return input_array

    def show_frame(self):
        frame = cv2.resize(self.stream.read(), (TARGET_WIDTH, TARGET_HEIGHT))
        frame = self.label_frame(frame)
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF

    def label_frame(self, frame):
        global label
        frame = cv2.putText(
            frame,
            str(label),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            TEXT_COLOUR,
            5,
            cv2.LINE_AA,
        )
        return frame


"""
Class to run inference over frames from the webcam
"""


class Inference:
    def __init__(self, camera):
        # init the camera
        self.camera = camera

    def infer(self):
        global label
        global model_ak
        while True:
            input_array = self.camera.get_input_array()
            predictions = model_ak.predict(input_array, num_classes=NUM_CLASSES)
            label = predictions[0]
            time.sleep(1 / INFERENCE_PER_SECOND)


label = 0
# create a model if one doesnt exist
if not os.path.exists(MODEL_FBZ):
    print("Initialising Akida model")
    initialise()

# load the akida model
model_ak = Model(filename=MODEL_FBZ)

camera = Camera()
controls = Controls(camera)
inference = Inference(camera)

# run inference in separate thread
t1 = threading.Thread(target=inference.infer)
t1.start()

# main loop to display camera feed, key capture and labels
while True:
    controls.capture()
