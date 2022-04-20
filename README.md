# Object_detection_YOLOv5

Model Name: YOLOv5

Project git hub: 

Model link: https://github.com/ultralytics/yolov5.git

About the model:
The model belongs to the family of compound-scaled object detection models trained on the COCO dataset. The model consist of three parts. (1) Backbone: CSPDarknet (2) Neck: PANet, and (3) Head: Yolo Layer. The data are first input to CSPDarknet for feature extraction, and then fed to PANet for feature fusion. Finally, Yolo Layer outputs detection results (class, score, location, size).

Primary analysis

Stages of Yolov5 model:
1.	Dataloader: class is used to iterate through the data, returning a tuple with the batch being passed through a deep neural network.
2.	Loss function: lets know the model of its inability to fit the data, with idea being to converge on an optimum set of parameters.
3.	Splitting into training and testing data: The issue may then be that the model "**overfits**" the training data and may fail when generalizing to a different subset. So we need for separate training, validation and testing steps which help combat overfitting. They help us have the idea of how well our model does on unseen data. In an effort to increase model's performance on validation data, we can tune training hyperparameters, model architecture and make use of data augmentation techniques.
4.	The metrics used to determine model performance is Mean Average Precision. Precision, which is the measure of the percentage of correctly predicted labels, and recall, which is the measure of how well the model was able to fit the datapoints corresponding to the positive class, are along with IoU (Intersection over Union) which is the area of the overlap between our predictions and ground truth. A threshold is usually chosen to classify whether the prediction is a true positive or a false negative. Average precision is the area under the precision-recall curve and follows precision and recall in having a value between 0 and 1. Interpolation of the precision value for a recall by the maximum precision which makes the curve between precision and recall be less susceptible to small changes in ranking of the points. Mean Average Precision (or mAP)** is calculated by average precision values for each class label.

Assumption:

lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0

Image Size = 1024 (32*32)
•	Convert format of input images and bounding boxes to the one supported by YOLOv5.
•	Load images and bouding boxes to the one supported by YOLOv5
•	Extract relevant field from images and annotations
•	Drop irrelevant columns from images and bouding box annotations
•	Splitting dataset to train and valuation subset.
•	Convert bounding box coordinates to the format supported by YOLOv5
•	Rescale bounding boxes 


False Positives:

The Model on the valuation set produced following metrics:
On All classes: 
Precision : 91.5%
Recall: 82%
mAP_0.5: 89.5%

On Person Class:
Precision: 90.6%
Recall: 79.5%
mAP_0.5: 87.3%

On Car Class:
Precision: 92.5%
Recall: 84.4%
mAP_0.5: 91.8%

Conclusion:	

The model is fast and has predicted with high precision and recall. 

Recommendation: 

Retraining with very low learning rate. This will increase the accuracy metrics and also training with more valuation set will give better results. 
