# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

def align(shape_predictor, imagePath):
  # initialize dlib's face detector (HOG-based) and then create
  # the facial landmark predictor
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(shape_predictor)
  fa = FaceAligner(predictor, desiredFaceWidth=256)

  # load the input image, resize it, and convert it to grayscale
  image = cv2.imread(imagePath)
  image = imutils.resize(image, width=500)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  # detect faces in the grayscale image
  rects = detector(gray, 1)

  # loop over the face detections
  for (i, rect) in enumerate(rects):
    # extract the ROI of the *original* face, then align the face
    # using facial landmarks
    (x, y, w, h) = rect_to_bb(rect)
    faceAligned = fa.align(image, gray, rect)

    return faceAligned