# Tasks for Deep_Learning_Assignment1_2_LinClassify.ipynb
#
# 1. Dataset Preparation:
#    - Load BloodMNIST dataset
#    - Preprocess the dataset (reshape, normalize)
#    - Split dataset into training, validation, and test sets
#
# 2. Implement SVM Loss:
#    - Implement svm_loss function to calculate hinge loss
#    - Include L2 regularization in the loss calculation
#    - Test the loss function with and without regularization
#
# 3. Gradient Calculation:
#    - Extend svm_loss to compute gradients for weight optimization
#
# 4. Implement Softmax Loss:
#    - Implement softmax_loss function with gradient calculation
#
# 5. Linear Classifier Class:
#    - Implement LinearClassifier class with methods for training and prediction
#    - Handle both SVM and Softmax loss types
#
# 6. Train the Classifier:
#    - Train the classifier and plot training history
#    - Optimize SVM classifier with hyperparameter search
#
# 7. Visualization:
#    - Visualize learned weights
#
# 8. Optimize Softmax Classifier:
#    - Repeat optimization steps for Softmax classifier
#
# 9. Documentation and Reporting:
#    - Document implementation with code snippets and results
#    - Discuss overfitting and include test set results with metrics

import medmnist
from medmnist import BloodMNIST
import numpy as np

# Load the BloodMNIST dataset
train_dataset = BloodMNIST(split="train", download=True, size=28)
val_dataset = BloodMNIST(split="val", download=True, size=28)
test_dataset = BloodMNIST(split="test", download=True, size=28)

# Extract images and labels
train_images, train_labels = train_dataset.imgs, train_dataset.labels
val_images, val_labels = val_dataset.imgs, val_dataset.labels
test_images, test_labels = test_dataset.imgs, test_dataset.labels

# Preprocess the dataset
# Reshape the images to vectors
train_images = train_images.reshape(train_images.shape[0], -1)
val_images = val_images.reshape(val_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)

# Normalize the images
train_images = train_images.astype(np.float32) / 255.0
val_images = val_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# Print dataset shapes
print(f'Train images shape: {train_images.shape}')
print(f'Validation images shape: {val_images.shape}')
print(f'Test images shape: {test_images.shape}')
