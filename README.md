# Food101 Image Classification

A deep learning project using a Convolutional Neural Network (CNN) to classify food images from the Food-101 dataset using PyTorch.

## Overview
This project demonstrates how CNNs can be applied to multi-class image classification problems in computer vision.

The goal is to explore feature extraction, training workflows, and model evaluation using a real-world image dataset.

## What This Project Does
- Loads and preprocesses images from the Food-101 dataset
- Trains a CNN model using PyTorch
- Evaluates model performance on validation data
- Demonstrates multi-class classification (101 food categories)

## Technologies Used
- Python
- PyTorch
- Torchvision
- CNNs
- Food-101 Dataset

## Why This Matters
Image classification is a core task in computer vision with applications in:
- Recommendation systems
- Automated labeling
- Food and nutrition analysis
- AI-powered visual understanding

This project reflects hands-on experience with deep learning pipelines and model training.

## How to Run

Install dependencies:
```bash
pip install -r requirements.txt

Train a CNN baseline on Food-101:

python train.py --epochs 3 --batch_size 32

Run inference on one image:

python predict.py --image path/to/your_image.jpg --ckpt checkpoints/food101_cnn.pt



Notes

Uses a simple CNN baseline (PyTorch) with standard image transforms.
Saves the best checkpoint based on test accuracy.



