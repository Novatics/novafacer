from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import pickle
import numpy as np
import cv2
import time
import imutils
import datetime
from PIL import Image
from identify_faces import load_image_file, identify_face
from extract_embeddings import extract_with_dlib

def get_greetings_text(name):
  now = datetime.datetime.now()
  if now.time() < datetime.time(12):
    return "Bom dia {}!".format(name)
  if now.time() < datetime.time(18):
    return "Boa tarde {}!".format(name)
  return "Boa noite {}!".format(name)

def use_image(image_path):
  test_image = load_image_file(image_path)
  recognizer = pickle.loads(open('output/recognizer.pickle', "rb").read())
  le = pickle.loads(open('output/le.pickle', "rb").read())

  face_locations = identify_face(test_image)
  no = len(face_locations)

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

def use_video():
  recognizer = pickle.loads(open('output/recognizer.pickle', "rb").read())
  le = pickle.loads(open('output/le.pickle', "rb").read())

  print("[INFO] starting video stream...")
  vs = VideoStream(src=0).start()
  time.sleep(2.0)

  fps = FPS().start()

  while True:
    frame = vs.read()

    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    recognizer = pickle.loads(open('output/recognizer.pickle', "rb").read())
    le = pickle.loads(open('output/le.pickle', "rb").read())

    face_locations = identify_face(frame)
    no = len(face_locations)

    test_image_encodings = extract_with_dlib(frame, face_locations)
    for i in range(no):
      test_image_encoding = test_image_encodings[i]
      class_probabilities = recognizer.predict_proba([test_image_encoding])
      j = np.argmax(class_probabilities[0])
      proba = class_probabilities[0][j]
      name = le.classes_[j]
      top, right, bottom, left = face_locations[i]
      text = "{}: {:.2f}%".format(name, proba * 100)
      greetings = get_greetings_text(name.capitalize())
      y = bottom - 10 if bottom - 10 > 10 else bottom + 10
      cv2.putText(frame, greetings, (10, (i + 1) * 35), cv2.FONT_HERSHEY_SIMPLEX, 1.20, (0, 0, 255), 2)
      cv2.putText(frame, text, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
      cv2.rectangle(frame, (left, bottom), (right, top), (0, 0, 255), 2)

    fps.update()

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
      break

  fps.stop()
  print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
  print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

  cv2.destroyAllWindows()
  vs.stop()

#######################################################################
#############################   Main   ################################
#######################################################################
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
	help="path to input image")
ap.add_argument("-f", "--format", required=False,
	help="recognition format")
args = vars(ap.parse_args())

if args["format"] == 'video':
  use_video()
else:
  use_image(args["image"])