import openface
import numpy as np
import PIL.Image
import sys
import dlib
import cv2
from skimage import io
from align_faces import align
from utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image, rect_to_tuple, trim_shape
from imutils import paths

face_detector = dlib.get_frontal_face_detector()

def load_image_file(file_path):
	im = PIL.Image.open(file_path)
	im = im.convert('RGB')
	return np.array(im)

def align_face():
	face_pose_predictor = dlib.shape_predictor(predictor_model)
	face_aligner = openface.AlignDlib(predictor_model)

def identify_face(img, number_of_times_to_upsample=1):
	return [trim_shape(rect_to_tuple(face), img.shape) for face in face_detector(img, number_of_times_to_upsample)]
