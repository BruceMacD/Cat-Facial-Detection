#! /usr/bin/env python
"""
Use the pre-trained Haar classifier from OpenCV to detect cat faces
"""

import sys
import cv2
from constants.constants import debug_cat_frontal_face_detection


# pre-trained classifier from OpenCV
CLASSIFIER = 'data/haarcascade_frontalcatface.xml'

detector = cv2.CascadeClassifier(CLASSIFIER)


def show_detected_faces(img, bounding_boxes):
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Detected cat faces: ", img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def detect_cat_face(img):
    # pre-process the image to grayscale, a normal step for haar classifiers
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # set a high scale factor because we care more about performance than accuracy
    # optional: use minNeighbors to decrease false positives by increasing threshold num bounding boxes
    # to be considered a face
    # use minSize to ignore faces that are too small, need to test different values here based on usage
    face_bounding_boxes = detector.detectMultiScale(grayscale_img, scaleFactor=1.7, minNeighbors=1, minSize=(75, 75))

    if debug_cat_frontal_face_detection:
        show_detected_faces(img, face_bounding_boxes)

    # TODO: find landmarks
    return face_bounding_boxes
