# Gender Classification Project

#### Website: [Gender Classification](https://gender-classification-wl8z.onrender.com/)
#### PowerPoint: [Presentation](https://docs.google.com/presentation/d/150DRxKDYMeYoaXfJIfI1T1bZK63FNpGSjWrkwybsdJY/edit)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Model Details](#model-details)
- [Conclusions](#conclusions)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Overview
This project aims to develop a gender classification system using machine learning techniques. The system takes an input image containing a human face and predicts the gender of the person in the image as either male or female.

## Features
- Uses state-of-the-art machine learning models for gender classification.
- Provides a user-friendly interface for easy interaction.
- Can be integrated into various applications such as security systems, demographic analysis tools, and more.

## Requirements
- Python
- TensorFlow (for machine learning models)
- NumPy
- Matplotlib (for visualization)
- Streamlit and Render (for web application)

## Model Details
The models implemented to classify genders and compare their performance on this task were Logistic Regression, Decision Tree, and Convolutional Neural Network (CNN). They were trained on a sample of the IMDB Wiki Faces Dataset. Key details include:

### Logistic Regression
- **Description:** A logistic regression model was used as a baseline for gender classification. It’s a simple yet effective linear model for binary classification tasks.
- **Performance Metrics:**
  - **Accuracy:** 56%
  - **Precision:** 56%
  - **Recall:** 56%

### Convolutional Neural Network (CNN)
- **Description:** A CNN was implemented to leverage its ability to capture spatial hierarchies in images. The architecture included convolutional layers, pooling layers, and fully connected layers.
- **Model Architecture:**
  - **Convolutional Layers:** Five convolutional layers of 64, 64, 128, 128, and 256 filters, respectively. Each with a kernel size of 3 or 5 and a ridge regularizer to reduce overfitting.
  - **Pooling Layers:** A 2x2 Max Pooling layer after every convolutional layer.
  - **Fully Connected Layers:** Two fully connected layers of 128 and 64 neurons, respectively, with a ReLU activation function, followed by a single-neuron output layer.
- **Performance Metrics:**
  - **Accuracy:** 93%
  - **Precision:** 93.5%
  - **Recall:** 93.5%

Each model was evaluated using standard performance metrics, and the CNN model demonstrated the best performance due to its ability to capture complex patterns in image data.

## Conclusions
The Convolutional Neural Network (CNN) model outperforms both Logistic Regression and Decision Tree Classifier in terms of precision, recall, F1-score, and accuracy on both the test and validation datasets. This indicates that the CNN model is better at capturing complex patterns in the data, making it the most promising model for gender classification. It achieves high accuracy and balanced performance metrics on both training and test datasets. However, it's worth noting that the CNN model's performance could have been even better. Babies tend to have very similar facial features regardless of gender, which may reduce the model's performance. Addressing this issue in future work could further enhance the model's accuracy and robustness.

## Contributing
Contributions are welcome! If you have any suggestions, bug reports, or want to contribute code, feel free to open an issue or submit a pull request.

## Acknowledgments
This project was inspired by similar projects in the field of computer vision and machine learning. We acknowledge the contributions of the open-source community for providing libraries and tools used in this project.
