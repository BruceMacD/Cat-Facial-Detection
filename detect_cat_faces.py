#! /usr/bin/env python
"""
Use the built-in functionality in OpenCV to detect cat faces.
"""

import sys
import getopt
import cv2
from components.cat_frontal_face_detection import detect_cat_face

EXPECTED_NUM_INPUTS = 1


def exit_error():
    print('Error: unexpected arguments')
    print('detect_cat_faces.py -i <path/to/inputCatImg.jpg>')
    sys.exit()


def main(argv):
    in_img = []
    try:
        opts, args = getopt.getopt(argv, "hi:", ["ifile="])
    except getopt.GetoptError:
        exit_error()

    # need specific number of ins
    if len(opts) != EXPECTED_NUM_INPUTS:
        exit_error()

    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            in_img = arg
        else:
            exit_error()

    print('Input file: ', in_img)

    cat_img = cv2.imread(in_img)

    detect_cat_face(cat_img)


if __name__ == "__main__":
    main(sys.argv[1:])
