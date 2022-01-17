# Reference from Kazuhito Takahashi(https://twitter.com/KzhtTkhs)
- [MediaPipe](https://mediapipe.dev/)
- [Kazuhito00/mediapipe-python-sample](https://github.com/Kazuhito00/mediapipe-python-sample)
- [Kazuhito00/hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)

# facial emotion recognition using mediapipe
- Estimate face mesh using MediaPipe(Python version).This is a sample program that recognizes facial emotion with a simple multilayer perceptron using the detected key points that returned from mediapipe.Although this model is 97% accurate, there is no generalization due to too little training data.
- the project is implement from https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe to use in facial emotion recognition
- the keypoint.csv is empty because this file is too large to upload so if you want to training model please find new dataset or record data by yourself

This repository contains the following contents.
- Sample program
- Facial emotion recognition model(TFLite)
- Script for collect data from images dataset and camera 

# Requirements
- mediapipe 0.8.9
- OpenCV 4.5.4 or Later
- Tensorflow 2.7.0 or Later
- scikit-learn 1.0.1 or Later (Only if you want to display the confusion matrix) 
- matplotlib 3.5.0 or Later (Only if you want to display the confusion matrix)

### main.py
This is a sample program for inference.it will use keypoint_classifier.tflite as model to predict your emotion.

### training.ipynb
This is a model training script for facial emotion recognition.

### model/keypoint_classifier
This directory stores files related to facial emotion recognition.
The following files are stored.
* Training data(keypoint.csv)
* Trained model(keypoint_classifier.tflite)
* Label data(keypoint_classifier_label.csv)
* Inference module(keypoint_classifier.py)

### Collect_from_image.py
This script will collect the keypoints from image dataset(.jpg). you can change your dataset directory to collect data.It will use your folder name to label.

### Collect_from_webcam.py
This script will collect the keypoints from your camera. press 'k' to enter the mode to save key points that show 'Record keypoints mode' then press '0-9' as label. the key points will be added to "model/keypoint_classifier/keypoint.csv". 

# Author
Rattasart Sakunrat

# License 
hand-gesture-recognition-using-mediapipe is under [Apache v2 license](LICENSE).