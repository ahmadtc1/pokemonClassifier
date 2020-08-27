from keras.models import load_model
import coremltools
import argparse
import pickle

#Make an argparser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-model", "--model", required=True, help="path to input model")
ap.add_argument("-l", "--labelbin", required=True, help="path to label binarizer")
args = vars(ap.parse_args())

print("[INFO] loading class labels from label binarizer...")
lb = pickle.load(open(args["labelbin"], "rb").read())
class_labels = lb.classes_.tolist()
print("[INFO] class labels: {}".format(class_labels))

#load trained cnn
print("[INFO] loading model...")
model = load_model(args["model"])

#convert model to coreml format
print("[INFO] converting model to coreml format...")
coreml_model = coremltools.converter.keras.convert(model,
    input_names="image",
    image_input_names="image",
    image_scale=1/255.0,
    class_labels=class_labels,
    is_bgr=True)

#Save model to disk
output = args["model"].rsplit(".", 1)[0] + ".mlmodel"
print("[INFO] saving model as {}".format(output))
coreml_model.save(output)