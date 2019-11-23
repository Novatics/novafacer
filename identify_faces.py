from imutils import paths
import sys
import dlib
import cv2
from skimage import io
from align_faces import align
from utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image
import openface

def identify(image_path, save_to_file = False):

	# Take the image file name from the command line
	file_name = image_path

	predictor_model = "shape_predictor_68_face_landmarks.dat"

	# Create a HOG face detector using the built-in dlib class
	face_detector = dlib.get_frontal_face_detector()
	face_pose_predictor = dlib.shape_predictor(predictor_model)
	face_aligner = openface.AlignDlib(predictor_model)

	#win = dlib.image_window()
	scale = 4
	# Load the image into an array
	image = cv2.imread(file_name)
	# img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
	# height, width = img.shape[:2]

	# Run the HOG face detector on the image data.
	# The result will be the bounding boxes of the faces in our image.
	# detected_faces = face_detector(image, 1)
	detected_faces, scores, idx = face_detector.run(image, 1)

	print("I found {} faces in the file {}".format(len(detected_faces), file_name))

	# Open a window on the desktop showing the image
	#win.set_image(image)
	# Loop through each face we found in the image
	for j, face_rect in enumerate(detected_faces):
		# Detected faces are returned as an object with the coordinates 
		# of the top, left, right and bottom edges
		if scores[j] > 0.75:
			# print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
			# Draw a box around each face we found
			# win.add_overlay(face_rect)

			pose_landmarks = face_pose_predictor(image, face_rect)
			aligned_face = face_aligner.align(534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
			####################################### METHOD 1 ####################################################
			# left_eye = extract_left_eye_center(shape)
			# right_eye = extract_right_eye_center(shape)
			# M = get_rotation_matrix(left_eye, right_eye)
			# rotated = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_CUBIC)
			# cropped = crop_image(img, face_rect)
			####################################### METHOD 1 ####################################################
			if save_to_file:
				cv2.imwrite("{}/{}/0000_{}.jpg".format('dataset/prepared', sys.argv[2], i), aligned_face)
				return True
			else
				return aligned_face
			# win.add_overlay(aligned_face)
			# return aligned_face
						
	# Wait until the user hits <enter> to close the window	        
	# dlib.hit_enter_to_continue()