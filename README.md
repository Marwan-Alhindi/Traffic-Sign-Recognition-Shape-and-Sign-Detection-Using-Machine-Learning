### Languages and Tools

- **Python**: For data processing, model training, and evaluation.
- **Pandas**: For handling data manipulation.
- **TensorFlow/Keras**: For building and training Convolutional Neural Networks.
- **OpenCV**: For image processing and manipulation.
- **Jupyter Notebook**: For development and running experiments.

# Traffic Sign Recognition: Shape and Sign Detection

## Overview

This project involves training and evaluating machine learning models to classify traffic signs based on two key tasks:
1. **Shape Detection**: Classifying traffic signs by their geometric shapes (e.g., round, triangle, rectangle).
2. **Sign Detection**: Classifying traffic signs by their specific types (e.g., stop, speed limit, yield).

The project utilizes a dataset consisting of European road traffic signs, provided in the form of grayscale images. The project implements supervised machine learning algorithms to classify the signs based on shape and sign type.

### Key Features

1. **Shape Detection**:
   - The first task is to classify images according to their shape. The images in the dataset are subdivided into categories such as round, triangle, rectangle, and others.
   - A supervised machine learning algorithm was trained to distinguish between these shapes.
   - Various models were evaluated to determine which best handled shape classification, including standard classifiers such as k-Nearest Neighbors (k-NN), Support Vector Machines (SVM), and Convolutional Neural Networks (CNN).

2. **Sign Detection**:
   - The second task involves identifying the specific type of traffic sign (e.g., stop, speed limit, yield).
   - Similar to the shape detection task, models were trained and evaluated to classify these sign types.
   - Advanced image processing techniques were used to handle variations in sign appearance due to environmental factors such as lighting, angles, and occlusion.

3. **Independent Evaluation**:
   - After training and testing the models on the provided dataset, independent evaluation was performed by collecting new images of traffic signs.
   - These new images were processed and fed into the trained models to evaluate their real-world performance.
   - The challenges faced in processing and classifying independently sourced data were documented and analyzed.

### System Design and Methodology

1. **Data Preprocessing**:
   - The images were preprocessed by resizing them to a uniform size of 28x28 pixels and converting them into grayscale to reduce computational complexity.
   - Feature scaling and normalization techniques were applied to ensure consistency across the dataset.
   
2. **Model Training**:
   - Supervised machine learning algorithms were used to train the models for both shape detection and sign detection.
   - The models evaluated include:
     - k-Nearest Neighbors (k-NN)
     - Support Vector Machines (SVM)
     - Convolutional Neural Networks (CNN)
   - Hyperparameter tuning and cross-validation were applied to improve model performance.

3. **Evaluation Metrics**:
   - Performance was evaluated using accuracy, precision, recall, and F1-score.
   - Cross-validation was used to assess the models' generalization ability and avoid overfitting.
   - The independent evaluation provided insights into how well the models perform on unseen data.

### Tasks and Results

1. **Shape Detection Task**:
   - Multiple algorithms were trained to classify traffic signs by shape. CNN outperformed traditional classifiers like k-NN and SVM, achieving the highest accuracy on both the training and test sets.
   
2. **Sign Detection Task**:
   - For sign-type classification, CNN models once again performed best, handling the complex task of recognizing various types of signs (e.g., speed limits, yield signs).
   
3. **Independent Testing**:
   - New traffic sign images were captured from real-world scenarios and processed for classification using the trained models. The models demonstrated robust performance, with CNN achieving high accuracy even on new, unseen data.
   - Challenges in independent evaluation included image quality, varying lighting conditions, and occlusions.

### Files in the Repository

- **`Traffic Sign Recognition - Shape Detection.ipynb`**: Jupyter Notebook for the shape detection task, including model training and evaluation.
- **`Traffic Sign Recognition - Sign Detection.ipynb`**: Jupyter Notebook for the sign detection task, including model training and evaluation.
- **`Traffic Sign Recognition - Shape and Sign Detection Testing.ipynb`**: Notebook for testing the models on independently collected traffic sign images.
- **`data.zip`**: The dataset used for training and testing.
- **`test_data.zip`**: Additional test data used for independent evaluation.
- **`models.zip`**: Saved models after training for both shape and sign detection.
- **`report.pdf`**: A detailed report discussing the methodologies, model performance, and findings.

### How to Run the Project

1. **Set Up Jupyter Notebook**:
   - Install Jupyter Notebook using Anaconda or any preferred method.
   - Ensure you have installed the required libraries such as `scikit-learn`, `TensorFlow/Keras`, and `OpenCV`.

2. **Run the Notebooks**:
   - Open the corresponding notebooks (`Shape Detection.ipynb`, `Sign Detection.ipynb`, and `Shape and Sign Detection Testing.ipynb`) to see the steps taken for model training, testing, and evaluation.
   - The notebooks include code for data preprocessing, model training, and generating evaluation metrics.

3. **Evaluate the Models**:
   - After training, evaluate the models on the independent test data provided in `test_data.zip`.
   - Use the `Traffic Sign Recognition - Shape and Sign Detection Testing.ipynb` to see how the models handle real-world traffic sign images.

### Data Source

The dataset consists of European road traffic signs, collected in real-world conditions. The dataset is divided into shape and sign-type categories, with images pre-processed to 28x28 pixel grayscale format.
