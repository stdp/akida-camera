from imutils.video import WebcamVideoStream
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array, smart_resize
import tensorflow_datasets as tfds
from akida_models import mobilenet_edge_imagenet_pretrained
from cnn2snn import convert
from akida import Model, FullyConnected


MODEL_FBZ = "models/edge_learning_example.fbz"

CAMERA_SRC = 0

NUM_NEURONS_PER_CLASS = 1
NUM_WEIGHTS = 350
NUM_CLASSES = 3

TARGET_WIDTH = 224
TARGET_HEIGHT = 224


class Camera:
    def __init__(self):
        self.stream = WebcamVideoStream(src=CAMERA_SRC).start()

    def get_frame(self):
        frame = self.stream.read()
        frame = smart_resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        return frame

    def get_input_array(self):
        frame = self.get_frame()
        input_array = img_to_array(frame)
        input_array = np.array([input_array], dtype="uint8")
        return input_array


class Inference:
    def __init__(self):
        self.camera = Camera()
        self.model_ak = Model(filename=MODEL_FBZ)

        # create a model if one doesnt exist
        if not os.path.exists(MODEL_FBZ):
            self.initialise()

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

    def learn(self, neuron):
        input_array = self.camera.get_input_array()
        self.model_ak.fit(input_array, neuron)

    def infer(self):
        input_array = self.camera.get_input_array()
        predictions = self.model_ak.predict(input_array, num_classes=NUM_CLASSES)
        print(predictions)


inference = Inference()

while True:
    inference.infer()
