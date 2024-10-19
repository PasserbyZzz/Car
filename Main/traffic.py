import cv2
import numpy as np
from new_driver import driver
import time
from threading import Thread
from Net import Net
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from detect import *

# Define a function to detect the color of a traffic light
def detect_traffic_light_color(frame):
    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for red, green, and yellow traffic lights
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    yellow_lower = np.array([11, 42, 144], np.uint8)
    yellow_upper = np.array([25, 255, 255], np.uint8)

    # Create masks for each color range
    red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)
    green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)

    # Count the number of pixels in each mask
    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)

    # Determine the dominant color based on pixel count
    dominant_color = None
    if red_pixels > green_pixels and red_pixels > yellow_pixels:
        dominant_color = "red"
    elif green_pixels > red_pixels and green_pixels > yellow_pixels:
        dominant_color = "green"
    elif yellow_pixels > red_pixels and yellow_pixels > green_pixels:
        dominant_color = "yellow"

    return dominant_color

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=20):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        #ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        #ret = self.stream.set(3,resolution[0])
        #ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True

# Initialize the video stream
videostream = VideoStream(resolution=(480,640),framerate=10).start()


# Main loop
while True:
    # Read the frame from the video stream
    frame_traffic = videostream.read()

    # Resize the frame for faster processing
    frame_traffic = cv2.resize(frame_traffic, (640, 480))

    # Detect the traffic light color
    traffic_light_color = detect_traffic_light_color(frame_traffic)

    # Control the car based on the traffic light color
    if traffic_light_color == "red":
        print("Traffic light is red. Stopping car.")
    elif traffic_light_color == "green":
        print("Traffic light is green. Moving car forward.")
    elif traffic_light_color == "yellow":
        print("Traffic light is yellow. Wait.")
    