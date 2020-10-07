from styx_msgs.msg import TrafficLight
import cv2
import numpy as np

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        # Convert image colour space from BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # red colour range in HSV space
        low_red = np.array([0, 150, 255])
        high_red = np.array([11, 255, 255])
        min_pixels_red = 52

        mask_red = cv2.inRange(hsv, low_red, high_red)
        nr_red_pixels = np.count_nonzero(mask_red)
        if nr_red_pixels > min_pixels_red:
            return TrafficLight.RED

        # yellow colour range in HSV space
        low_yellow = np.array([28, 150, 255])
        high_yellow = np.array([31, 255, 255])
        min_pixels_yellow = 52

        mask_yellow = cv2.inRange(hsv, low_yellow, high_yellow)
        nr_yellow_pixels = np.count_nonzero(mask_yellow)
        if nr_yellow_pixels > min_pixels_yellow:
            return TrafficLight.YELLOW

        # green colour range in HSV space
        low_green = np.array([36, 150, 255])
        high_green = np.array([76, 255, 255])
        min_pixels_green = 52

        mask_green = cv2.inRange(hsv, low_green, high_green)
        nr_green_pixels = np.count_nonzero(mask_green)
        if nr_green_pixels > min_pixels_green:
            return TrafficLight.GREEN

        return TrafficLight.UNKNOWN
