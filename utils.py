import tensorflow as tf
import numpy as np
import cv2
import time


'''
The purpose of this function is to perform non-maximum suppression (NMS) on the outputs of an object detection model.

The function takes in several arguments, including inputs, which contains the bounding box coordinates,
 confidence scores, and class probabilities for each object detected by the model, as well as model_size, max_output_size, max_output_size_per_class, iou_threshold, and confidence_threshold, which are parameters for controlling the behavior of the NMS algorithm.

The function first splits the inputs tensor into separate tensors for bounding box coordinates (bbox),
 confidence scores (confs), and class probabilities (class_probs). It then scales the bounding box coordinates to match the size of the input image.

Next, the function computes the detection scores by multiplying the confidence scores and class probabilities
 together. It then applies the combined_non_max_suppression function from TensorFlow to perform NMS on the detections. 
 This function selects the highest-scoring detections, discards any overlapping detections,
   and returns the remaining detections as boxes, scores, classes, and valid_detections tensors.

Finally, the function returns the boxes, scores, classes, and valid_detections tensors,
 which contain the output of the NMS algorithm. These tensors can be used to visualize and
   analyze the detections made by the object detection model.
'''


def non_max_suppression(inputs, model_size, max_output_size,
                        max_output_size_per_class, iou_threshold,
                        confidence_threshold):
    bbox, confs, class_probs = tf.split(inputs, [4, 1, -1], axis=-1)
    bbox = bbox/model_size[0]
    scores = confs * class_probs
    boxes, scores, classes, valid_detections = \
        tf.image.combined_non_max_suppression(
            boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
            scores=tf.reshape(scores, (tf.shape(scores)[0], -1,
                                       tf.shape(scores)[-1])),
            max_output_size_per_class=max_output_size_per_class,
            max_total_size=max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=confidence_threshold
        )
    return boxes, scores, classes, valid_detections




def resize_image(inputs, modelsize):
    inputs = tf.image.resize(inputs, modelsize)
    return inputs


def load_class_names(file_name):
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

'''
The purpose of the output_boxes function is to extract the bounding boxes of the detected objects from 
the model output and perform non-maximum suppression to remove duplicate boxes and select the most confident boxes.

The function takes in the model output inputs, model size, maximum output size,
 maximum output size per class, IOU threshold, and confidence threshold as input parameters.
   It then splits the inputs into separate tensors for the center x, center y, width, height,
     confidence, and classes.

Using these values, the function computes the top-left and bottom-right coordinates of the bounding boxes.
 It then concatenates these values with confidence scores and class predictions to form the inputs tensor.

Finally, the function calls the non_max_suppression function to perform non-maximum suppression
 on the inputs tensor and return the filtered bounding boxes in the form of a dictionary.
'''
def output_boxes(inputs, model_size, max_output_size, max_output_size_per_class,
                 iou_threshold, confidence_threshold):
    center_x, center_y, width, height, confidence, classes = \
        tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)
    top_left_x = center_x - width / 2.0
    top_left_y = center_y - height / 2.0
    bottom_right_x = center_x + width / 2.0
    bottom_right_y = center_y + height / 2.0
    inputs = tf.concat([top_left_x, top_left_y, bottom_right_x,
                        bottom_right_y, confidence, classes], axis=-1)
    boxes_dicts = non_max_suppression(inputs, model_size, max_output_size,
                                      max_output_size_per_class, iou_threshold, confidence_threshold)
    return boxes_dicts



'''
The purpose of the function draw_outputs is to draw bounding boxes and class labels on an input image based on the output of the YOLO model.

The function takes in the following parameters:

img: the input image
boxes: a numpy array of shape (N, 4) containing the coordinates of the bounding boxes for N objects detected in the image
objectness: a numpy array of shape (N,) containing the objectness score for each of the N objects detected
classes: a numpy array of shape (N,) containing the class label (index) for each of the N objects detected
nums: a numpy array of shape (1,) containing the number of objects detected in the image
class_names: a list of string containing the names of the classes
The function first converts the normalized coordinates of the bounding boxes back to pixel coordinates,
 then loops through all the detected objects and draws a rectangle with the corresponding class
   label and objectness score on the input image. Finally, the function returns the image with bounding boxes 
   and labels drawn on it.
'''

def draw_outputs(img, boxes, objectness, classes, nums, class_names):
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    boxes = np.array(boxes)
    for i in range(nums):
        x1y1 = tuple(
            (boxes[i, 0:2] * [img.shape[1], img.shape[0]]).astype(np.int32))
        x2y2 = tuple(
            (boxes[i, 2:4] * [img.shape[1], img.shape[0]]).astype(np.int32))
        img = cv2.rectangle(img, (x1y1), (x2y2), (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            (x1y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    return img
