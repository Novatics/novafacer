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

# def _rect_to_css(rect):
#     """
#     Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order
#     :param rect: a dlib 'rect' object
#     :return: a plain tuple representation of the rect in (top, right, bottom, left) order
#     """
#     return rect.top(), rect.right(), rect.bottom(), rect.left()

# def _trim_css_to_bounds(css, image_shape):
#     """
#     Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.
#     :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
#     :param image_shape: numpy shape of the image array
#     :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
#     """
#     return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

# def _raw_face_locations(img, number_of_times_to_upsample=1, model="hog"):
#     return face_detector(img, number_of_times_to_upsample)

def load_image_file(file_path):
	im = PIL.Image.open(file_path)
	im = im.convert('RGB')
	return np.array(im)

def align_face():
	face_pose_predictor = dlib.shape_predictor(predictor_model)
	face_aligner = openface.AlignDlib(predictor_model)

def identify_face(img, number_of_times_to_upsample=1):
	return [trim_shape(rect_to_tuple(face), img.shape) for face in face_detector(img, number_of_times_to_upsample)]

def identify(image_path, save_to_file = False):

	# Take the image file name from the command line
	file_name = image_path

	predictor_model = "shape_predictor_68_face_landmarks.dat"

	# Create a HOG face detector using the built-in dlib class
	face_detector = dlib.get_frontal_face_detector()
	face_pose_predictor = dlib.shape_predictor(predictor_model)
	face_aligner = openface.AlignDlib(predictor_model)

	scale = 4
	# Load the image into an array
	image = cv2.imread(file_name)
	# img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
	# height, width = img.shape[:2]

	# Run the HOG face detector on the image data.
	# The result will be the bounding boxes of the faces in our image.
	# detected_faces = face_detector(image, 1)
	detected_faces, scores, idx = face_detector.run(image, 1)

	# Open a window on the desktop showing the image
	#win.set_image(image)
	# Loop through each face we found in the image
	for j, face_rect in enumerate(detected_faces):
		# print("Detected {} faces in {} with score {}".format(len(detected_faces), file_name, *scores))
		# Detected faces are returned as an object with the coordinates 
		# of the top, left, right and bottom edges
		# if scores[j] > 0.75:
			# print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
			# Draw a box around each face we found
			# win.add_overlay(face_rect)
		landmarkMap = {
        'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
        'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
    }
		pose_landmarks = face_pose_predictor(image, face_rect)
		bb = face_aligner.getLargestFaceBoundingBox(image)
		aligned_face = face_aligner.align(534, image, bb, landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
		
		if save_to_file:
			cv2.imwrite("{}/{}/0000_{}.jpg".format('dataset/prepared', sys.argv[2], i), aligned_face)
			return True
		else:
			return aligned_face
		# win.add_overlay(aligned_face)
		# return aligned_face
						
	# Wait until the user hits <enter> to close the window	        
	# dlib.hit_enter_to_continue()