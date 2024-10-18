import os
from ultralytics import YOLO
from PIL import Image

# last updated Aug 16, 2024 by Lucas DeMello (Brady Lab)
# my email is lucasxbox288@gmail.com if you have questions
# this code uses a trained YOLOv8 model to predict bounding boxes.
# Every time you run it, the .json file with the predicted bounding boxes will be found in the following folder:

#   C:\Users\Root Project\runs\detect (select the most recent folder within this)
#   OR IN THIS ONE (idk why but it changes randomly for me)
#   C:\Users\Root Project\Documents\yolov2\runs\detect (select the most recent folder within this)

# go to "yolopredstococo.py" to convert that .json file to a usable COCO annotation file.
# it predicts on the validation set, so if you want to change the images it is predicting on, go to:

#   C:\Users\Root Project\Documents\yolov2\dataset\val

# and put the images you want to predict on in the "images" folder. Note that when training the model, if you want
# a set of annotated validation images to compute metrics, I have prepared that in the folders "images (pre-annotated)" and
# "labels (pre-annotated)", simply rename them "images" and "labels" and then run the training code. You can replace the images
# and annotations as needed, but just remember that is where they need to go if you want metrics for the training.

###################################################################
# Prediction code below
###################################################################

def main():
    # Load the trained model
    model_path = r'C:\Users\Root Project\Documents\yolov2\training_runs\train_yolov8\yolov8_endodermis3\weights\best.pt'
    model = YOLO(model_path)

    # Run validation and save results
    results = model.val(save=True, save_json=True, save_conf=True)
    
    print("Validation completed. Results should be saved in the model's project directory.")
    print(f"mAP50: {results.box.map50}")
    print(f"mAP50-95: {results.box.map}")

if __name__ == '__main__':
    main()