import os
from ultralytics import YOLO

# last updated Aug 16, 2024 by Lucas DeMello (Brady Lab)
# my email is lucasxbox288@gmail.com if you have questions
# this code trains a model using rectangle annotations in YOLO format.
# to edit the input training images/annotations or validation images/annotations,
# go to the .yaml file listed below. There are no inputs necessary in this file.
# Note that when training the model, if you want
# a set of annotated validation images to compute metrics, I have prepared that in the folders "images (pre-annotated)" and
# "labels (pre-annotated)", simply rename them "images" and "labels" and then run the training code. You can replace the images
# and annotations as needed, but just remember that is where they need to go if you want metrics for the training.
# It still works without this step.

###################################################################
# Training code below
###################################################################

def main():
    # Paths and Parameters
    data_yaml = r'C:\Users\Root Project\Documents\yolov2\dataset.yaml'
    model_output_dir = r'C:\Users\Root Project\Documents\yolov2\training_runs\train_yolov8'
    pretrained_weights = 'yolov8n.pt'

    # Create output directory if it does not exist
    os.makedirs(model_output_dir, exist_ok=True)

    # Load YOLOv8 model
    model = YOLO(pretrained_weights)

    # Train the model
    model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        name='yolov8_endodermis',
        project=model_output_dir,
        patience=20,
        save=True,
        device='0'  # Use GPU if available, otherwise CPU
    )

if __name__ == '__main__':
    main()