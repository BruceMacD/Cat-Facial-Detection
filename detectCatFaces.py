#! /usr/bin/env python
"""
Use the built-in functionality in OpenCV to detect cat faces.
"""

import sys
import getopt
import cv2

EXPECTED_NUM_INPUTS = 1


def exit_error():
    print('Error: unexpected arguments')
    print('detectCatFaces.py -i <path/to/inputCatImg.jpg>')
    sys.exit()


def detect_cat_face(img):
    # TODO
    return


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

    cv2.imshow("Cat image: ", cat_img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1:])
