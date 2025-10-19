# YOLOv8 Person and Car Detection - Key Insights

## Project Summary

This project demonstrates the implementation of a YOLOv8 object detection model to detect persons and cars in images. The model was trained for 60 epochs and achieved an mAP50 of 0.653 and mAP50-95 of 0.376 on the validation set.

## Key Results

### Performance Metrics

After 60 epochs of training, the model achieved:

- **mAP50**: 0.653
- **mAP50-95**: 0.376
- **Precision**: ~0.73
- **Recall**: ~0.60

These results indicate that the model has learned to effectively detect both persons and cars, with particularly strong performance at the 0.5 IoU threshold.

### Training Progress

![Training Results](runs/detect/train2/results.png)
*Training metrics over 60 epochs showing the model's learning progress.*

### Precision-Recall Curve

![PR Curve](runs/detect/train2/PR_curve.png)
*Precision-Recall curve showing the trade-off between precision and recall for both classes.*

### Confusion Matrix

![Confusion Matrix](runs/detect/train2/confusion_matrix.png)
*Confusion matrix showing correct classifications vs. misclassifications.*

## Model Performance Analysis

### Strengths

1. **Good mAP50 Score**: The model achieves a solid 0.653 mAP50, indicating good detection accuracy at the 0.5 IoU threshold.

2. **Balanced Class Performance**: The model performs reasonably well on both classes (person and car) as shown in the PR curve.

3. **Effective Transfer Learning**: Starting with COCO-pretrained weights helped accelerate training and improve performance.

### Areas for Improvement

1. **Recall**: The recall (~0.60) suggests that the model misses about 40% of objects, which could be improved.

2. **mAP50-95**: The mAP50-95 of 0.376 indicates room for improvement in precise bounding box localization.

3. **Class Imbalance**: If one class is underrepresented in the dataset, performance on that class might suffer.

## Sample Predictions

The following images show example predictions on test data:

![Prediction 1](runs/detect/final_predictions/image_000000034_jpg.rf.7a04a279594485c2a0d1f1487b427cf5.jpg)
*Example prediction showing detected persons and cars with bounding boxes.*

![Prediction 2](runs/detect/final_predictions/image_000000045_jpg.rf.1cf2ed6be9c43756838374df242a3c84.jpg)
*Another example showing the model's detection capabilities.*

![Prediction 3](runs/detect/final_predictions/image_000000113_jpg.rf.fa4070b3fcd6b4e7210c0d6d3a7dd395.jpg)
*Example demonstrating the model's performance on more complex scenes.*

## Technical Details

### Model Architecture

- **Base Model**: YOLOv8s (small version)
- **Pretrained Weights**: COCO dataset
- **Input Size**: 640x640 pixels
- **Batch Size**: 16

### Training Configuration

- **Epochs**: 60
- **Optimizer**: Auto (typically SGD with momentum)
- **Learning Rate**: 0.01 with decay
- **Augmentation**: Applied during training

### Dataset

- **Classes**: 2 (person, car)
- **Total Images**: 2,243
- **Format**: YOLO format with bounding boxes
- **Preprocessing**: Images resized to 416x416 pixels

## Conclusion

The YOLOv8 model successfully learned to detect persons and cars in images with reasonable accuracy. The project demonstrates a complete computer vision pipeline from data preparation to model training, evaluation, and inference.

The model's performance is solid for many practical applications, though there's room for improvement, particularly in recall and precise localization. Future work could involve experimenting with larger model variants, hyperparameter tuning, or more advanced data augmentation techniques.

## Recommendations for Improvement

1. **Increase Training Time**: Train for more epochs to see if performance plateaus or continues to improve.

2. **Experiment with Model Sizes**: Try YOLOv8m or YOLOv8l for potentially better performance at the cost of speed.

3. **Hyperparameter Tuning**: Fine-tune learning rate, augmentation parameters, and other hyperparameters.

4. **Address Class Imbalance**: If one class is underrepresented, consider data augmentation or weighted loss functions.

5. **Advanced Augmentation**: Implement more sophisticated augmentation techniques like Mosaic or MixUp.