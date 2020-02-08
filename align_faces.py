# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import openface
from utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image

def align(detected_face_rect, imagePath):
  # initialize dlib's face detector (HOG-based) and then create
  # the facial landmark predictor
  shape_predictor = 'shape_predictor_68_face_landmarks.dat'
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(shape_predictor)
  fa = FaceAligner(predictor, desiredFaceWidth=256)
  face_aligner = openface.AlignDlib(shape_predictor)

  # load the input image, resize it, and convert it to grayscale
  image = cv2.imread(imagePath)
  image = imutils.resize(image, width=500)
  (s_height, s_width) = image.shape[:2]
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  

  # detect faces in the grayscale image
  # rects = detected_face_rect

  # loop over the face detections
  # for (i, rect) in enumerate(rects):
    # extract the ROI of the *original* face, then align the face
    # using facial landmarks
    
  (x, y, w, h) = rect_to_bb(detected_face_rect)
  faceAligned = fa.align(image, gray, detected_face_rect)

    #pose_landmarks = predictor(gray, rect)
    #faceAligned = face_aligner.align(534, gray, rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

    #shape = predictor(gray, rect)
    #left_eye = extract_left_eye_center(shape)
    #right_eye = extract_right_eye_center(shape)
    #M = get_rotation_matrix(left_eye, right_eye)
    #rotated = cv2.warpAffine(gray, M, (s_width, s_height), flags=cv2.INTER_CUBIC)

    #faceAligned = crop_image(rotated, rect)


  return faceAligned


""" ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
args = vars(ap.parse_args())

faceAligned = align('shape_predictor_68_face_landmarks.dat', args["dataset"])
cv2.imshow("Aligned", faceAligned)
cv2.waitKey(0) """