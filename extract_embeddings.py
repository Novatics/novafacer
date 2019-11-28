from imutils import paths
from align_faces import align
from utils import tuple_to_rect
import numpy as np
import argparse
import imutils
import pickle
import cv2
import dlib
import os
import sys

pose_predictor_68_point = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
pose_predictor_5_point = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def face_landmarks(face_image, face_locations=None, model="large"):
	face_locations = [tuple_to_rect(face_location) for face_location in face_locations]
	pose_predictor = pose_predictor_68_point

	if model == "small":
			pose_predictor = pose_predictor_5_point

	return [pose_predictor(face_image, face_location) for face_location in face_locations]

def extract_with_dlib(face_image, known_face_locations, num_jitters=1):
	raw_landmarks = face_landmarks(face_image, known_face_locations, model="large")
	face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
	return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]


def extract(face):
	embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')
	faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
	embedder.setInput(faceBlob)
	vec = embedder.forward()
	return vec.flatten()
