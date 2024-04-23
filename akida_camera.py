import os
import cv2
import threading
import time
import queue
from pynput import keyboard
from picamera2 import Picamera2
from rpi_ws281x import PixelStrip, Color

from akida import Model as AkidaModel, devices, AkidaUnsupervised, FullyConnected
from akida_models import akidanet_edge_imagenet_pretrained
from cnn2snn import convert, set_akida_version, AkidaVersion

import numpy as np
from tensorflow.image import resize_with_crop_or_pad

LED_PIN = 18
NUM_LEDS = 8

CAMERA_SRC = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

TEXT_COLOUR = (190, 30, 255)

MODEL_FBZ = "models/edge_learning_example.fbz"

NEURON_KEYS = [str(i) for i in range(10)]
SAVE_BUTTON = "s"

COLOURS = {
    0: Color(0, 0, 0),
    1: Color(255, 0, 0),
    2: Color(255, 128, 0),
    3: Color(255, 255, 0),
    4: Color(0, 255, 0),
    5: Color(0, 255, 255),
    6: Color(0, 0, 255),
    7: Color(255, 0, 255),
}

class Camera:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=30)
        self.camera = Picamera2()
        self.stream_config = self.camera.create_video_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
        self.camera.configure(self.stream_config)
        self.camera.start()
        self.running = True
        self.shots = {}

        threading.Thread(target=self.capture_frames, daemon=True).start()
        threading.Thread(target=self.show_window, daemon=True).start()

    def capture_frames(self):
        while self.running:
            frame = self.camera.capture_array()
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

    def get_frame(self):
        return self.frame_queue.get()

    def get_input_array(self, target_width, target_height):
        frame = self.frame_queue.get()
        if frame is not None:
            processed_frame = self.process_frame(frame, target_width, target_height)
            return processed_frame

    def show_window(self):
        while self.running:
            frame = self.get_frame()
            cv2.imshow("AkidaCamera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.running = False

    def process_frame(self, frame, target_width, target_height):
        if frame is not None:
            frame = resize_with_crop_or_pad(frame, target_width, target_height)
            expanded_array = np.expand_dims(frame, axis=0)
            int8_array = expanded_array.astype("uint8")
            return int8_array


class WS2812Controller:
    def __init__(self, pin, num_leds):
        """
        Initialize the WS2812 RGB LED controller.

        Args:
        pin (int): The GPIO pin connected to the data input of the LEDs.
        num_leds (int): Number of LEDs in the strip.
        """
        self.colours = COLOURS
        self.num_leds = num_leds
        self.strip = PixelStrip(num_leds, pin)
        self.strip.begin()
        self.cleanup()

    def show_colour(self, index):
        self.cleanup()
        self.strip.setPixelColor(index, self.colours[index])
        self.strip.show()

    def cleanup(self):
        """
        Clean up by turning off all LEDs.
        """
        for i in range(self.num_leds):
            self.strip.setPixelColor(i, Color(0, 0, 0))
        self.strip.show()


class Controls:

    """
    Class to capture key presses to save/learn
    """

    def __init__(self, inference):
        self.listener = keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        )
        self.listener.start()
        self.inference = inference

    def on_press(self, key):
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


class Inference:

    def __init__(self):

        # create a new model if one doesnt exist
        if not os.path.exists(MODEL_FBZ):
            print("Initialising Akida model")
            self.initialise()

        self.camera = Camera()
        self.controls = Controls(self)
        self.lights = WS2812Controller(LED_PIN, NUM_LEDS)
        self.saved = []
        self.labels = {}

        # load the akida model
        self.model_ak = AkidaModel(filename=MODEL_FBZ)

        if len(devices()) > 0:
            device = devices()[0]
            self.model_ak.map(device)

        # run inference in separate thread
        threading.Thread(target=self.infer).start()

    def initialise(self):

        """
        Method to initialise an Akida model if one doesn't exist
        """

        with set_akida_version(AkidaVersion.v1):
            model_keras = akidanet_edge_imagenet_pretrained()

        # convert it to an Akida model
        model_ak = convert(model_keras)

        # Replace the last layer by a classification layer
        num_classes = 8
        num_neurons_per_class = 1
        num_weights = 350
        model_ak.pop_layer()
        layer_fc = FullyConnected(name='akida_edge_layer',
                                  units=num_classes * num_neurons_per_class,
                                  activation=False)
        model_ak.add(layer_fc)
        model_ak.compile(
            optimizer=AkidaUnsupervised(
                num_weights=num_weights,
                num_classes=num_classes,
                learning_competition=0.1
            )
        )

        # save new model
        model_ak.save(MODEL_FBZ)

    def infer(self):
        while True:
            input_array = self.camera.get_input_array(224, 224)
            predictions = self.model_ak.predict_classes(input_array, num_classes=8)
            predicted_class = int(predictions[0])
            if predicted_class in self.saved:
                self.lights.show_colour(predicted_class)
            #time.sleep(0.0001)

    def learn(self, neuron):
        input_array = self.camera.get_input_array(224, 224)
        self.model_ak.fit(input_array, neuron)
        if neuron not in self.saved:
            self.saved.append(neuron)
            self.camera.shots[neuron] = 1
        else:
            self.camera.shots[neuron] += 1
        self.camera.label = "Learned {}".format(self.labels.get(neuron, neuron))

    def save(self):
        self.model_ak.save(MODEL_FBZ)


def main():
    try:
        # Initialize the Sentry object
        inference = Inference()
        print("Inference system is running.")

        # The system will keep running in the background, you can add more logic here if needed
        # For example, running some tests or adding user interaction
        time.sleep(
            600
        )  # Keep the main thread alive for 10 minutes or until Ctrl+C is pressed

    except Exception as e:
        print(f"Failed to start Inference system: {e}")

    finally:
        pass


if __name__ == "__main__":
    main()
