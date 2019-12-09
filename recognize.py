import argparse
import pickle
import numpy as np
import cv2
from PIL import Image
from identify_faces import load_image_file, identify_face
from extract_embeddings import extract_with_dlib

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

test_image = load_image_file(args["image"])
recognizer = pickle.loads(open('output/recognizer.pickle', "rb").read())
le = pickle.loads(open('output/le.pickle', "rb").read())

face_locations = identify_face(test_image)
no = len(face_locations)
print("Number of faces detected: ", no)

test_image_encodings = extract_with_dlib(test_image, face_locations)
for i in range(no):
  test_image_encoding = test_image_encodings[i]
  class_probabilities = recognizer.predict_proba([test_image_encoding])
  j = np.argmax(class_probabilities[0])
  proba = class_probabilities[0][j]
  name = le.classes_[j]
  top, right, bottom, left = face_locations[i]
  text = "{}: {:.2f}%".format(name, proba * 100)
  y = bottom - 10 if bottom - 10 > 10 else bottom + 10
  cv2.putText(test_image, text, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
  cv2.rectangle(test_image, (left, bottom), (right, top), (0, 0, 255), 2)

pil_image = Image.fromarray(test_image)
pil_image.resize((100,100))
pil_image.show()