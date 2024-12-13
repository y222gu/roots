from ultralytics import YOLO
import os

if __name__ == '__main__':
    path_to_yaml = os.path.join(os.getcwd(),"aerenchyma_segmentation", "yolo.yaml")

    # Load a pre-trained YOLOv8 model
    model = YOLO('yolo11m-seg.pt')  # Replace with 'yolov8s.pt', 'yolov8m.pt', etc., for larger models

    # Train the model on your dataset
    model.train(
        data= path_to_yaml, # Path to your dataset YAML file
        epochs=400,                # Number of epochs
        batch=16,                 # Batch size
        imgsz=1024,                # Image size
        project=r'aerenchyma_segmentation\runs',     # Save training results to project/name
        name='2024_12_11_yolov11_segmentation',     # Experiment name
        workers=4,
        task='segmentation'               # Number of workers for data loading
    )

    # Evaluate the model
    model.val()

    model.export(format='onnx')  # Options: 'onnx', 'torchscript', 'coreml', etc.
