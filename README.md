# Handwritten Digit Recognition (MNIST)
**Author: Le Hoang Duong (Hanoi University of Science and Technology)**

## Project Overview
This project implements a Multilayer Perceptron (MLP) to classify handwritten digits from the MNIST dataset. It was developed as a practical application of the Machine Learning Specialization by Stanford University and DeepLearning.AI.

The primary focus is ensuring numerical stability in loss calculations and performing model diagnostics through Cross-Validation (CV) analysis.

## Model Architecture
The network is designed with a sequential 3-layer structure to process 28 x 28 grayscale images:

* **Input Layer**: Accepts 28 x 28 pixel grayscale images.
* **Flatten Layer**: Converts the 2D image matrix into a 1D array of 784 units.
* **Hidden Layer 1**: Dense layer with 25 neurons using ReLU activation.
* **Hidden Layer 2**: Dense layer with 15 neurons using ReLU activation.
* **Output Layer**: Dense layer with 10 neurons (representing digits 0-9) using Linear activation (Logits).

**Numerical Stability**: I implemented the `from_logits=True` parameter within the Sparse Categorical Cross-Entropy loss function. This technique improves numerical precision by combining the Softmax activation and loss calculation into a single operation.

## Performance and Analysis
The model achieved a test accuracy of approximately 97% after 10 epochs of training.

### Cross-Validation Analysis
To diagnose model performance, I utilized a 20% validation split to monitor for Overfitting (High Variance):

* **Training vs. CV Loss**: The convergence of both training and validation loss indicates a well-fitted model with minimal variance.
* **Accuracy Stability**: Consistent performance across training and validation sets confirms that the model generalizes effectively to unseen handwriting data.

## Technology Stack
* **Framework**: TensorFlow / Keras
* **Data Processing**: NumPy
* **Visualization**: Matplotlib
* **IDE**: Visual Studio Code
