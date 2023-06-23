# MNIST Digit Classification CNN Model

This repository contains a Convolutional Neural Network (CNN) model trained on the MNIST dataset for digit classification. The MNIST dataset is a widely-used benchmark dataset in the field of machine learning, consisting of 60,000 training images and 10,000 test images of handwritten digits from 0 to 9.

The purpose of this model is to classify the input images of handwritten digits into their respective categories using a simple CNN architecture. The model has been trained and tested using the popular deep learning framework, and it achieves a high accuracy on the MNIST dataset.


## Model Training

The model has been trained on the MNIST dataset using the following steps:

1. Data preprocessing: The MNIST dataset images are normalized by dividing each pixel value by 255 to obtain values between 0 and 1. Additionally, the labels are one-hot encoded to represent each class as a binary vector.

2. Model initialization: The CNN model with the specified architecture is created.

3. Model training: The model is trained using the training set images and labels. The training process involves forward propagation, computing the loss, backward propagation, and weight updates. The training continues for a specified number of epochs or until convergence.

4. Model evaluation: The trained model is evaluated using the test set images and labels. The accuracy metric is calculated to measure the performance of the model on unseen data.

## handwritten dataset
![image](https://github.com/keerthikkn/MNIST_Degit_Classification/assets/42544473/860365c3-f8c2-4c5e-9038-e10729add79c)

## model performance
![image](https://github.com/keerthikkn/MNIST_Degit_Classification/assets/42544473/20737bd9-7228-498f-8c50-7a6f7a79ee34)


## Model Usage

To use the trained model for digit classification, follow these steps:

1. Install the required dependencies mentioned in the `requirements.txt` file.

2. Load the model by training the model from scratch.

3. Preprocess the input image to match the required format (appropriate size).

4. Feed the preprocessed image to the model and obtain the predicted probabilities for each digit class.

5. Choose the class with the highest probability as the predicted digit label.

An example usage code snippet is provided in the repository to guide you through the process.

## Repository Contents

- `degit_classification.ipynb`: Contains the implementation of the CNN model architecture, training, and evaluation procedures.
- `requirements.txt`: Lists the required dependencies for running the code.
- `train.csv` : training dataset.
- `test.csv` : test dataset.

## Conclusion

The provided CNN model demonstrates accurate digit classification on the MNIST dataset. Feel free to use this model for your own digit recognition tasks or as a starting point for further experimentation and research in the field of deep learning.

Please refer to the original source or consult the documentation for more detailed information about the implementation and usage of the model.
