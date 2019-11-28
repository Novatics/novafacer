from PIL import Image
import argparse
import os
import pickle
import cv2
import imutils
import numpy as np
from imutils import paths
from identify_faces import identify, load_image_file, identify_face
from extract_embeddings import extract, extract_with_dlib
from train_model import train
from sklearn import svm

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
args = vars(ap.parse_args())

knownEmbeddings = []
knownNames = []

image_paths = list(paths.list_images(args["dataset"]))
for (i, image_path) in enumerate(image_paths):
  name = image_path.split(os.path.sep)[-2]
  face = load_image_file(image_path)
  face_bounding_boxes = identify_face(face)
  if face_bounding_boxes and len(face_bounding_boxes) > 0:
    print("Adding embeddings for {}".format(name))
    face_enc = extract_with_dlib(face, face_bounding_boxes)[0]
    knownEmbeddings.append(face_enc)
    knownNames.append(name)

clf = svm.SVC(gamma='scale', probability=True)
clf.fit(knownEmbeddings, knownNames)

test_image = load_image_file('images/raianne_3.jpg')
# image = cv2.imread('images/cecilia_2.jpg')
# image = imutils.resize(image, width=600)

face_locations = identify_face(test_image)
no = len(face_locations)
print("Number of faces detected: ", no)

print("Found:")
test_image_encodings = extract_with_dlib(test_image, face_locations)
for i in range(no):
  test_image_enc = test_image_encodings[i]
  # name = clf.predict([test_image_enc])
  class_probabilities = clf.predict_proba([test_image_enc])
  j = np.argmax(class_probabilities[0])
  proba = class_probabilities[0][j]
  name = clf.classes_[j]
  top, right, bottom, left = face_locations[i]
  text = "{}: {:.2f}%".format(name, proba * 100)
  y = bottom - 10 if bottom - 10 > 10 else bottom + 10
  cv2.putText(test_image, text, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
  cv2.rectangle(test_image, (left, bottom), (right, top), (0, 0, 255), 2)
  # cv2.rectangle(image, (left, bottom), (right, top), (0, 0, 255), 2)
  # text = "{}: {:.2f}%".format(name, proba * 100)
  # y = bottom - 10 if bottom - 10 > 10 else bottom + 10
  # cv2.putText(image, text, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


# test_image_enc = extract_with_dlib(test_image, face_locations)[0]
# class_probabilities = clf.predict_proba([test_image_enc])
# for i, face_location in face_locations:
#   class_probabilities[i]
#   top, right, bottom, left = face_location
#   cv2.rectangle(test_image, (left, bottom), (right, top),
# 			(0, 0, 255), 2)

# cv2.imshow("Image", image)
# cv2.waitKey(0)

# face_image = test_image[top:bottom, left:right]
pil_image = Image.fromarray(test_image)
pil_image.show()