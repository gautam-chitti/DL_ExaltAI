# YOLOv8 Object Detection for Person and Car Detection

This project implements a complete pipeline for training and evaluating a YOLOv8 object detection model to detect two classes of objects: 'person' and 'car'. The model is trained on a custom dataset and achieves an mAP50 of 0.653 and mAP50-95 of 0.376.

## Project Overview

This project demonstrates how to:
- Train a YOLOv8 object detection model on a custom dataset
- Evaluate model performance using standard metrics
- Visualize training results and predictions
- Package the complete project for reproducibility

The final model is capable of detecting and localizing persons and cars in images with good accuracy.

## Dataset

The dataset used in this project contains 2,243 images with person and car annotations in YOLO format. The dataset is split into train, validation, and test sets.

- **Classes**: 2 (person, car)
- **Image size**: Resized to 416x416 pixels during preprocessing
- **Annotations**: YOLO format with bounding boxes

## Model Architecture

We use the YOLOv8s (small) model, which provides a good balance between performance and computational cost. The model is initialized with weights pre-trained on the COCO dataset to leverage transfer learning.

## Training Configuration

- **Epochs**: 60
- **Image size**: 640x640 pixels
- **Batch size**: 16
- **Optimizer**: Auto (typically SGD with momentum)
- **Learning rate**: 0.01 (with decay)
- **Data augmentation**: Applied during training

## Results

After training for 60 epochs, the model achieved the following performance metrics on the validation set:

- **mAP50**: 0.653
- **mAP50-95**: 0.376
- **Precision**: ~0.73
- **Recall**: ~0.60

The results show that the model has learned to effectively detect both persons and cars in images, with room for improvement in recall.

## Directory Structure

```
.
├── data/
│   ├── train/
│   ├── val/
│   ├── test/
│   └── data.yaml
├── runs/
│   ├── detect/
│   │   ├── train2/
│   │   │   ├── weights/
│   │   │   │   ├── best.pt
│   │   │   │   └── last.pt
│   │   │   ├── results.png
│   │   │   ├── confusion_matrix.png
│   │   │   └── PR_curve.png
│   │   └── final_predictions/
└── Notebook.ipynb
```

## Key Files

- `Notebook.ipynb`: Main Jupyter notebook containing the complete training and evaluation pipeline
- `data/data.yaml`: Dataset configuration file
- `runs/detect/train2/weights/best.pt`: Best performing model weights
- `runs/detect/train2/results.png`: Training metrics visualization
- `runs/detect/train2/confusion_matrix.png`: Model performance analysis
- `runs/detect/final_predictions/`: Directory containing test set predictions

## Usage

To use this project:

1. Ensure you have the required dependencies installed:
   ```
   pip install ultralytics==8.2.103
   ```

2. The main notebook (`Notebook.ipynb`) contains the complete pipeline:
   - Data preparation
   - Model training
   - Evaluation
   - Prediction on test set
   - Result visualization

3. To run inference with the trained model:
   ```bash
   yolo task=detect mode=predict model=runs/detect/train2/weights/best.pt source=/path/to/your/images
   ```

## Visual Results

Example predictions on test images show the model's ability to detect persons and cars in various scenarios. Check the `runs/detect/final_predictions/` directory for sample outputs.

## Future Improvements

Potential areas for improvement:
- Increase training epochs
- Experiment with different model sizes (YOLOv8m, YOLOv8l)
- Fine-tune hyperparameters
- Apply more advanced data augmentation techniques
- Address class imbalance if present