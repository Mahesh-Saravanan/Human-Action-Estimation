# Human Action Estimation using OpenPose Data

## Overview

This project is designed to estimate actions performed by an individual based on the movement of body parts referenced by x, y coordinates at each time step. The data is sourced from OpenPose, providing detailed information about body parts such as shoulders, arms, legs, and knees. The project explores five distinct machine learning approaches to accomplish accurate action estimation.

## Machine Learning Approaches

1. **Convolutional Neural Network 1D (CNN_1d.py):** Training pipeline for 1D CNN.
2. **Convolutional Neural Network 2D (CNN_2d.py):** Training pipeline for 2D CNN.
3. **Recurrent Neural Network (LSTM) (RNN_.py):** Training pipeline for LSTM-based model.
4. **Random Forest (rf.py):** Random Forest implementation.
5. **Dynamic Time Warping (DTW.py):** Implementation of Dynamic Time Warping for data processing and fitting.
6. **Principle Component Analysis (Clustering) (pca_lsd_.py):** Code for PCA-based clustering.

## Results

After experimenting with the different models, the Recurrent Neural Network (LSTM) was identified as the best-performing model among the options.

## Usage

The following files are provided for usage and exploration:

- `CNN_1d.py`: Training pipeline for 1D CNN.
- `CNN_2d.py`: Training pipeline for 2D CNN.
- `DTW.py`: Dynamic Time Warping (data processing and fitting).
- `RNN_.py`: Training pipeline for LSTM.
- `rf.py`: Random Forest implementation.
- `dataload.py`: Code for data loading and preprocessing.
- `pca_lsd_.py`: Principle component analysis.
- `compare_models.py`: Code to validate the performance of different models.
- `visualization.py`: Code for visualizing results.
- `Presentation.ipynb`: Executable Jupyter notebook file encompassing all methods.

The folder `trained models` holds the trained models for deep learning methods.

## Contributors

- Christeena Varghese
-  Fatima Mohamed
-  Mahesh Saravanan
-  Muhammad Zakriya Shah Sarwar
-  Nehmiya Shikur
