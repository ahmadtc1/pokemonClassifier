import matplotlib as plt
plt.use("Agg")

#import necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from model.smallVggNet import SmallVggNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to dataset")
ap.add_argument("-m", "--model", required=True, help="path to the model")
ap.add_argument("-l", "--labelbin", required=True, help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png", 
    help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

#Initialize num of epochs, initial learning rate, batch size and image dimensions
EPOCHS = 100
INIT_LR = 1e-3
BS = 32
IMAGE_DIMENSIONS=(96, 96, 3)

#Init data and labels
data = []
labels = []
#Obtain image paths and shuffle them randomly
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(11)
random.shuffle(imagePaths)

#Loop over the input images
for imagePath in imagePaths:
    #load the image, apply some pre-processing, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMENSIONS[1], IMAGE_DIMENSIONS[0]))
    image = img_to_array(image)
    data.append(image)

    #Obtain the class label using the path and update the labels list
    #file structure is dataset/{CLASS_LABEL}/{FILENAME}.jpg
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

#Scale the raw pixel intensities to within the range [0, 1]
#All the values are within the range [0, 255] so divide by 255.0
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))

#Binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

#Partition data into training and testing using 80/20 split respectively
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=11)


#Construct image generator for data augmentation 
#Data augmentation allows us to train our model on augmented images based off our existing 
# images for a more accurate trained model
aug = ImageDataGenerator(rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=2,
    fill_mode="nearest")

#initialize the model
print("[INFO] compiling model...")
model = SmallVggNet.build(width=IMAGE_DIMENSIONS[1], height=IMAGE_DIMENSIONS[0],
    depth=IMAGE_DIMENSIONS[2], classes=len(lb.classes_))
optimizer = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=optimizer,
    metrics=["accuracy"])

#train the network
print("[INFO] training network...")
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1)

#save the model to disk
print("[INFO] serializing network...")
model.save(args["model"], save_format="h5")

#Save the label binarizer to disk...
print("[INO] serializing label binarizer..")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()

# plot training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])
