import argparse
import os
import pickle
from imutils import paths
from identify_faces import identify_face, load_image_file
from extract_embeddings import extract, extract_with_dlib
from train_model import train, train_svm

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to training dataset")
args = vars(ap.parse_args())

knownEmbeddings = []
knownNames = []

image_paths = list(paths.list_images(args["dataset"]))
for (i, image_path) in enumerate(image_paths):
  name = image_path.split(os.path.sep)[-2]
  face = load_image_file(image_path)
  indentified_face = identify_face(face)
  if indentified_face is not None and len(indentified_face) > 0:
    print("Generating embeddings vector for {}".format(name))
    face_encodings = extract_with_dlib(face, indentified_face)[0]
    knownEmbeddings.append(face_encodings)
    knownNames.append(name)

train_svm(knownEmbeddings, knownNames)