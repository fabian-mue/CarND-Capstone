from styx_msgs.msg import TrafficLight
import tensorflow as tf
import cv2
import numpy as np

class TLClassifier(object):
    def __init__(self):
        self.threshold = 0.51
        self.use_simulator = True

        # Load frozen inference graph
        if self.use_simulator:
            trained_model_path = 'light_classification/ssd_simulator/frozen_inference_graph_sim.pb'
        else:
            trained_model_path = 'light_classification/ssd_real/frozen_inference_graph_real.pb'
        self.ssd_graph = tf.Graph()
        with self.ssd_graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(trained_model_path, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')

            # Load Graph and extract relevant tensors reflecting interesting inputs and outputs
            self.image_tensor = self.ssd_graph.get_tensor_by_name('image_tensor:0')
            self.boxes_det = self.ssd_graph.get_tensor_by_name('detection_boxes:0')
            self.scores_det = self.ssd_graph.get_tensor_by_name('detection_scores:0')
            self.classes_det = self.ssd_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections_det = self.ssd_graph.get_tensor_by_name('num_detections:0')

        # set current session as attribute
        self.current_session = tf.Session(graph=self.ssd_graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # TODO implement light color prediction
        with self.graph.as_default():
            image_expanded = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
            # Start detection
            (boxes_sess, scores_sess, classes_sess, num_detections_sess) = self.current_session.run(
                [self.boxes_det, self.scores_det, self.classes_det, self.num_detections_det],
                feed_dict={self.image_tensor: image_expanded})

        # Remove not required dimensions
        scores_pp = np.squeeze(scores_sess)
        classes_pp = np.squeeze(classes_sess).astype(np.int32)

        if scores_pp[0] > self.threshold:
            if 1 == classes_pp[0]:
                print('Green light')
                return TrafficLight.GREEN
            elif 2 == classes_pp[0]:
                print('Red light')
                return TrafficLight.RED
            elif 3 == classes_pp[0]:
                print('Yellow light')
                return TrafficLight.YELLOW
        return TrafficLight.UNKNOWN

    def get_classification_cv(self, image):
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
