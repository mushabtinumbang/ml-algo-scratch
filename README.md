# ml-algo-scratch

This project provides machine learning algorithms implemented from scratch.

---

## Prerequisites

Make sure conda is installed.

---

## Conda Environment

### Installation

If you have **Make** installed, you can run this command.
```bash
$ make create-env
```
Else, you can run this command.
```bash
$ conda env update --file environment.yml
```

---

## Running Linear Regression

The Linear Regression CLI entrypoint is `src/main/main_linear_regression.py`. Use the provided `run_linreg.sh` script or Makefile target with environment variables.

### Using Shell Script (Recommended)

Edit parameters in `run_linreg.sh`, then run:
```bash
$ ./run_linreg.sh
```

### Using Makefile

Set environment variables and run the Makefile target.

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| `N_SAMPLES` | `1000` | Number of samples to generate. |
| `N_FEATURES` | `1` | Number of features in the synthetic dataset. |
| `NOISE` | `20.0` | Noise level added to the regression targets. |
| `RANDOM_STATE` | `42` | Random seed for reproducibility. |
| `TEST_SIZE` | `0.2` | Fraction of data used for the test set. |
| `LEARNING_RATE` | `0.01` | Gradient descent learning rate. |
| `N_ITERS` | `1000` | Number of gradient descent iterations. |
| `OPTIMIZATION_METRIC` | `mse` | Optimization metric: `mse`, `mae`, or `r2`. |
| `VERBOSE` | `""` | Set to `--verbose` to print training progress. |

#### Sample Pipeline Command: Linear Regression

To run with default parameters:
```bash
$ make run-linear-regression
```

To run with custom parameters:
```bash
$ export N_SAMPLES=2000 &&
export N_FEATURES=5 &&
export LEARNING_RATE=0.05 &&
export OPTIMIZATION_METRIC=mae &&
make run-linear-regression
```

To run with verbose output:
```bash
$ export VERBOSE="--verbose" &&
make run-linear-regression
```

---

## Linear Regression Output Artifacts

All plots are saved to the image path configured in `src/utilities/config_.py` (`img_path`). By default, this is the `img/` directory in the project root.

| File | Condition | Description |
| :--- | :--- | :--- |
| `loss_history.jpg` | Always | Training loss over iterations. |
| `metrics_history.jpg` | Always | MSE/MAE/R2 over iterations. |
| `predictions.jpg` | Only when `N_FEATURES == 1` | Actual vs Predicted plot. |

---

## Running KNN Classifier

The KNN Classifier CLI entrypoint is `src/main/main_knn.py`. Use the provided `run_knn.sh` script or Makefile target with environment variables.

### Using Shell Script (Recommended)

Edit parameters in `run_knn.sh`, then run:
```bash
$ ./run_knn.sh
```

### Using Makefile

Set environment variables and run the Makefile target.

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| `N_SAMPLES` | `1000` | Number of samples to generate. |
| `N_FEATURES` | `2` | Number of features in the synthetic dataset. |
| `N_CLASSES` | `2` | Number of classes in the classification task. |
| `N_INFORMATIVE` | `2` | Number of informative features. |
| `RANDOM_STATE` | `42` | Random seed for reproducibility. |
| `TEST_SIZE` | `0.2` | Fraction of data used for the test set. |
| `K` | `3` | Number of nearest neighbors. |
| `VERBOSE` | `""` | Set to `--verbose` to print training progress. |

#### Sample Pipeline Command: KNN Classifier

To run with default parameters:
```bash
$ make run-knn
```

To run with custom parameters:
```bash
$ export N_SAMPLES=2000 &&
export N_FEATURES=3 &&
export N_CLASSES=3 &&
export K=5 &&
make run-knn
```

To run with verbose output:
```bash
$ export VERBOSE="--verbose" &&
make run-knn
```

# ml-algo-scratch
