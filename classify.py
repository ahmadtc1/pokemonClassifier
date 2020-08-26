#import packages
# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained model")
ap.add_argument("-l", "--labelbin", required=True, help="path to label binarizer")
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

#Load image
image = cv2.imread(args["image"])
output = image.copy()

#pre process for classification
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

#Load trained cnn and label binarizer
print("[INFO] loading cnn...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

#Classify input image
print("[INFO] classifying image...")
probability = model.predict(image)[0]
idx = np.argmax(probability)
label = lb.classes_[idx]

#Mark our prediction as "correct" if the filename of the input image matches the 
# predicted text label (assuming test image files named this way)
filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
correct = "correct" if filename.rfind(label) != -1 else "incorrect"

#build the label and draw label on the img
label = "{}: {:.2f}% ({})".format(label, probability[idx] * 100, correct)
output = imutils.resize(output, width = 400)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

#Show the output image
print("[INFO] {}".format(label))
cv2.imshow("classified", output)
cv2.waitKey(0)