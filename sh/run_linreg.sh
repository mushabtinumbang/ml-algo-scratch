#!/bin/bash

# Linear Regression Run Script
# Set parameters below, then run: ./run_linreg.sh

# Data Parameters
export N_SAMPLES=10000
export N_FEATURES=5
export NOISE=20.0
export RANDOM_STATE=42
export TEST_SIZE=0.2

# Model Parameters
export LEARNING_RATE=0.01
export N_ITERS=10000
export OPTIMIZATION_METRIC="mse"

# Verbose Mode (set to "--verbose" to enable, empty string to disable)
export VERBOSE="--verbose"

# Run Linear Regression
make run-linear-regression
