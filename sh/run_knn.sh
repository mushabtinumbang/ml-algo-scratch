#!/bin/bash

# KNN Classifier Run Script
# Set parameters below, then run: ./run_knn.sh

# Data Parameters
export N_SAMPLES=50000
export N_FEATURES=4
export N_CLASSES=2
export N_INFORMATIVE=2
export RANDOM_STATE=42
export TEST_SIZE=0.2

# Model Parameters
export K=3

# Verbose Mode (set to "--verbose" to enable, empty string to disable)
export VERBOSE=""

# Run KNN Classifier
make run-knn
