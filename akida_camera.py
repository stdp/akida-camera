from imutils.video import VideoStream
import numpy as np
import os
import cv2
from threading import Thread
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow_datasets as tfds

from akida_models import mobilenet_edge_imagenet_pretrained
from cnn2snn import convert
from akida import Model, FullyConnected

import time

MODEL_FBZ = "models/edge_learning_example.fbz"

CAMERA_SRC = 0
CAMERA_FPS = 30

NUM_NEURONS_PER_CLASS = 1
NUM_WEIGHTS = 350
NUM_CLASSES = 3

TARGET_WIDTH = 224
TARGET_HEIGHT = 224


class Controls:
    def __init__(self, inference):
        self.name = "keyboard_controls"
        self.inference = inference

    def capture(self):
        if cv2.waitKey(33) == ord("1"):
            print("learn 0")
            self.inference.learn(0)

        if cv2.waitKey(33) == ord("2"):
            print("learn 1")
            self.inference.learn(1)

        if cv2.waitKey(33) == ord("3"):
            print("learn 2")
            self.inference.learn(2)

        if cv2.waitKey(33) == ord("s"):
            print("saving model")
            self.inference.save()


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
        frame = self.get_frame()
        input_array = img_to_array(frame)
        input_array = np.array([input_array], dtype="uint8")
        return input_array

    def show_frame(self):
        frame = self.get_frame()
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF


class Inference:
    def __init__(self, camera):
        # init the camera
        self.camera = camera
        # create a model if one doesnt exist
        if not os.path.exists(MODEL_FBZ):
            print("Initialising Akida model")
            self.initialise()
        # load the akida model
        self.model_ak = Model(filename=MODEL_FBZ)

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
        model_ak.add(layer_fc)
        model_ak.compile(
            num_weights=NUM_WEIGHTS, num_classes=NUM_CLASSES, learning_competition=0.1
        )
        model_file = os.path.join("", MODEL_FBZ)
        model_ak.save(model_file)

    def learn(self, neuron):
        input_array = self.camera.get_input_array()
        self.model_ak.fit(input_array, neuron)

    def save(self):
        model_file = os.path.join("", MODEL_FBZ)
        self.model_ak.save(model_file)

    def infer(self):
        input_array = self.camera.get_input_array()
        predictions = self.model_ak.predict(input_array, num_classes=NUM_CLASSES)
        print(predictions[0])


camera = Camera()
inference = Inference(camera)
controls = Controls(inference)

while True:
    controls.capture()
    camera.show_frame()
    inference.infer()
    time.sleep(1 / CAMERA_FPS)
