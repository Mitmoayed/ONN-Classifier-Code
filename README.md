
# ONN Classifier Code

## Description
This repository contains the implementation of an Oscillatory Neural Network (ONN) 
classifier for handwritten digit recognition using the MNIST dataset. 
The system uses a single-layer network of 36 coupled oscillators governed by the 
Kuramoto model, with class prototypes optimized using a Genetic Algorithm (GA) 
and coupling weights established through Hebbian learning.

## Requirements
- Python 3.x
- NumPy
- SciPy
- OpenCV (cv2)
- TensorFlow/Keras (for MNIST loading)
- scikit-learn
- matplotlib
- pandas
- numba
- joblib
- geneticalgorithm

Install dependencies:
pip install numpy scipy opencv-python tensorflow scikit-learn matplotlib pandas numba joblib geneticalgorithm

## Data Preprocessing
Each 28×28 MNIST image is processed as follows:
1. Center crop to 24×24
2. Resize to 6×6 using OpenCV
3. Binarize using global threshold
4. Map to binary values {0, 1}
5. Convert to phase values {-π/2, +π/2}

## Genetic Algorithm Parameters
- Population size: 30
- Maximum generations: 100
- Crossover probability: 0.5
- Mutation probability: 0.06
- Elite ratio: 0.5
- Tournament size: 4
- Fitness function: classification accuracy from Kuramoto simulations

## Usage
1. Place the pre-optimized prototype file `bestsloution1ndpart1.csv` 
   in the same directory as the script
2. Run the main script:
python test10class_edit1.py

## Pre-optimized Prototype Patterns
The optimized prototype patterns for all 45 digit pairs are provided 
in `bestsloution1ndpart1.csv`. These patterns were obtained by running 
the GA optimization and can be used directly for inference without 
re-running the optimization.

## Results

- 10-class MNIST accuracy: 75-76%

