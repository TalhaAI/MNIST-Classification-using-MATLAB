# MNIST Classification Project

This repository contains the project for the course "Computer Methods in Chemical Engineering." The project involves the classification of the MNIST dataset using different classifiers like Multi-Layer Perceptron (MLP), k-nearest neighbors (KNN), and Convolutional Neural Network (CNN) with the predefined functions available in MATLAB.

## Table of Contents

- [Introduction](#introduction)
- [Methods](#methods)
  - [Multi-Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
  - [K-nearest neighbor (KNN)](#k-nearest-neighbor-knn)
  - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)

## Introduction

We are performing classification of the MNIST dataset using different classifiers such as Multi-Layer Perceptron (MLP), k-nearest neighbors (KNN), and Convolutional Neural Network (CNN). The MNIST dataset is a collection of 60,000 training images and 10,000 testing images of handwritten digits (0-9).

## Methods

### Multi-Layer Perceptron (MLP)

- Load the MNIST dataset into MATLAB using CSV files.
- Normalize the input data.
- Convert labels to one-hot encoding.
- Initialize weights and biases.
- Train the model using stochastic gradient descent with backpropagation.
- Test the model on the test set and compute the confusion matrix.

### K-nearest neighbor (KNN)

- Load the MNIST dataset into MATLAB using CSV files.
- Normalize the input data.
- Train the KNN classifier.
- Test the model on the test set and compute the confusion matrix.

### Convolutional Neural Network (CNN)

- Use the deep learning toolbox in MATLAB to create a CNN.
- Define the network architecture with convolutional layers, max-pooling layers, and fully connected layers.
- Train the model on the MNIST dataset.
- Test the model on the test set and compute the confusion matrix.
