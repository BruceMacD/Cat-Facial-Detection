#! /usr/bin/env python
"""
Use the pre-trained Haar classifier from OpenCV to detect cat faces
"""

import cv2
import dlib
import numpy as np
from constants.constants import debug_cat_frontal_face_detection


# pre-trained classifier from OpenCV
CLASSIFIER = 'data/haarcascade_frontalcatface.xml'
# Pre-trained shape predictor from iBUG 300-W dataset for human facial landmarks
SHAPE_PREDICTOR = 'data/shape_predictor_68_face_landmarks.dat'

detector = cv2.CascadeClassifier(CLASSIFIER)
landmarks_predictor = dlib.shape_predictor(SHAPE_PREDICTOR)


def show_detected_faces(img, bounding_boxes, facial_landmark_points):
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # draw circles for landmarks
    for x, y in facial_landmark_points:
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Detected cat faces: ", img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


# conversion from imutils
def landmarks_to_numpy(landmarks):
    # initialize the matrix of (x, y)-coordinates with a row for each landmark
    coords = np.zeros((landmarks.num_parts, 2), dtype=int)

    # convert each landmark to (x, y)
    for i in range(0, landmarks.num_parts):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

    # return the array of (x, y)-coordinates
    return coords


def detect_cat_face(img):
    facial_landmark_points = []

    # pre-process the image to grayscale, a normal step for haar classifiers
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # set a high scale factor because we care more about performance than accuracy
    # optional: use minNeighbors to decrease false positives by increasing threshold num bounding boxes
    # to be considered a face
    # use minSize to ignore faces that are too small, need to test different values here based on usage
    face_bounding_boxes = detector.detectMultiScale(grayscale_img, scaleFactor=1.7, minNeighbors=1, minSize=(75, 75))

    # TODO: currently testing a human facial landmark detector on the cat faces,
    #       will probably need to train a new landmark detector
    for face in face_bounding_boxes:
        # TODO: need to convert face to the form: rectangles[[(241, 170) (562, 491)]]
        landmarks = landmarks_predictor(img, face)
        facial_landmark_points.append(landmarks_to_numpy(landmarks))

    if debug_cat_frontal_face_detection:
        show_detected_faces(img, face_bounding_boxes, facial_landmark_points)

    return face_bounding_boxes
