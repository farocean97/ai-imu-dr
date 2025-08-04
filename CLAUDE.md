# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains the implementation of AI-IMU Dead-Reckoning, a novel method for accurately estimating vehicle position and orientation using only an Inertial Measurement Unit (IMU). The approach combines:

1. An Invariant Extended Kalman Filter (IEKF) that integrates IMU measurements
2. A deep learning-based adapter that dynamically optimizes filter noise parameters

The system achieves 1.10% translational error on the KITTI odometry dataset using only IMU data.

## Code Architecture

The codebase is organized into several key components:

1. **Data Processing**
   - `dataset.py` - Base dataset class and KITTI dataset implementation for handling IMU data
   - `utils.py` - Utility functions for data preparation and manipulation

2. **Kalman Filter Implementation**
   - `utils_numpy_filter.py` - NumPy implementation of the IEKF for evaluation
   - `utils_torch_filter.py` - PyTorch implementation of the IEKF for training
   - The filter integrates IMU measurements with zero lateral and vertical velocity constraints

3. **Deep Learning Components**
   - `train_torch_filter.py` - Training pipeline for the filter adapter network
   - Neural networks in `utils_torch_filter.py` that dynamically adapt noise parameters

4. **Evaluation and Visualization**
   - `utils_plot.py` - Functions for visualizing results
   - `main_kitti.py` - Main script to run the system on KITTI data

## Common Commands

### Setup and Installation

```bash
# Install required packages
pip install matplotlib numpy termcolor scipy navpy torch
```

### Data Preparation

```bash
# Download and extract KITTI IMU data
wget "https://github.com/user-attachments/files/17930695/data.zip"
mkdir results
unzip data.zip
rm data.zip

# Download pre-trained model parameters
wget "https://www.dropbox.com/s/77kq4s7ziyvsrmi/temp.zip"
unzip temp.zip -d temp
rm temp.zip
```

### Running the System

```bash
# Run the model on KITTI data
cd src
python3 main_kitti.py
```

### Training a New Model

To train the model with different settings:

```bash
# Modify KITTIArgs in main_kitti.py:
# - Set train_filter = 1
# - Set test_filter = 0 if you only want to train

cd src
python3 main_kitti.py
```

## Key Parameters

The filter parameters are defined in the `KITTIParameters` class in `main_kitti.py`:

- `g`: Gravity vector
- `cov_*`: Process noise covariance parameters
- `cov_*0`: Initial state covariance parameters

The training parameters are defined in `train_torch_filter.py`:

- `lr_*`: Learning rates for different network components
- `weight_decay_*`: Weight decay settings for optimizer
- `max_loss`, `max_grad_norm`: Training stability parameters

## Data Flow

1. IMU data is loaded and preprocessed from KITTI dataset
2. The network adapter converts raw IMU signals into covariance matrices
3. The IEKF uses these covariance matrices to integrate IMU measurements
4. The filter outputs position, velocity, and orientation estimates
5. For training, a custom loss function compares position estimates to ground truth

## Evaluation

The system achieves 1.10% translational error on the KITTI odometry dataset. The code includes functions for calculating relative pose errors and visualizing trajectories.