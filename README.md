# PRODIGY_ML_03

# Cat and Dog Image Classification using HOG and SVM

This project aims to classify images of cats and dogs using the Histogram of Oriented Gradients (HOG) feature descriptor and the Support Vector Machine (SVM) classifier.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Preprocessing](#data-preprocessing)
3. [Model Training](#model-training)
4. [Model Evaluation](#model-evaluation)
5. [Prediction](#prediction)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [Installation and Usage](#installation-and-usage)
9. [Dependencies](#dependencies)

## Introduction

This project demonstrates the use of the Histogram of Oriented Gradients (HOG) feature descriptor in combination with a Support Vector Machine (SVM) classifier to distinguish between images of cats and dogs. It covers the full pipeline from data preprocessing to model training, evaluation, and prediction.

## Data Preprocessing

1. **Mount Google Drive:**
    - Mount Google Drive to access the image dataset.

2. **Load and Resize Images:**
    - Iterate through the dataset directory structure, loading and resizing each image to a uniform size.

3. **Extract HOG Features:**
    - Extract the HOG feature descriptor from each image. HOG captures the edge and gradient structure, which is particularly useful for distinguishing between different classes of images.

4. **Flatten HOG Features:**
    - Flatten the HOG features into a single vector for each image to create a feature matrix.

5. **Split Data:**
    - Split the dataset into training and testing sets to evaluate the performance of the classifier.

## Model Training

1. **Grid Search for Hyperparameters:**
    - Perform a grid search to find the optimal hyperparameters for the SVM classifier, such as the regularization parameter and kernel type.

2. **Train SVM Classifier:**
    - Train the SVM classifier using the training data with the best-found hyperparameters.

## Model Evaluation

1. **Evaluate Classifier:**
    - Evaluate the performance of the trained SVM classifier on the testing data.
  
2. **Report Accuracy:**
    - Report the accuracy of the classifier. In this project, the classifier achieved an accuracy of 95% on the testing data.

## Prediction

1. **User Input:**
    - Prompt the user to enter the path to an image for classification.

2. **Preprocess Image:**
    - Preprocess the image using the same steps as during training: resize and extract HOG features.

3. **Predict Class:**
    - Use the trained SVM classifier to predict whether the image is of a cat or a dog.

## Results

- The project achieved an accuracy of 95% on the testing data.
- The classifier can correctly classify both cats and dogs with high accuracy.

## Conclusion

- This project demonstrates the effectiveness of using HOG features in combination with an SVM classifier for image classification tasks.
- The approach can be generalized to classify other types of images with minor modifications.



## Dependencies

- Python 3.x
- OpenCV
- scikit-image
- scikit-learn
- NumPy
- Google Colab (optional, if running on Google Drive)

