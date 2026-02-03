import numpy as np
import click
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.features.knn import KNNClassifier
from src.utilities.config_ import log_path
from src.utilities.utils import setup_logger
from loguru import logger

# log
setup_logger("knn", log_path)

@click.command()
@click.option('--n_samples', type=int, default=1000, help='Number of samples')
@click.option('--n_features', type=int, default=2, help='Number of features')
@click.option('--n_classes', type=int, default=2, help='Number of classes')
@click.option('--n_informative', type=int, default=2, help='Number of informative features')
@click.option('--random_state', type=int, default=42, help='Random state for reproducibility')
@click.option('--test_size', type=float, default=0.2, help='Test set size')
@click.option('--k', type=int, default=3, help='Number of neighbors')
@click.option('--verbose', is_flag=True, help='Print training progress')
def main(n_samples, n_features, n_classes, n_informative, random_state, test_size, k, verbose):
    """
    KNN classifier from scratch.
    """
    logger.info("Starting to run KNN Classifier ...")

    logger.info(
        "Data Generation Params- \n"
        + f" Samples: {n_samples} |\n"
        + f" Features: {n_features} |\n"
        + f" Classes: {n_classes} |\n"
        + f" Informative features: {n_informative} |\n"
        + f" Random state: {random_state} |\n"
    )

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_informative,
        random_state=random_state
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("=" * 80)
    print(
        "Split Results\n"
        + f" Training samples: {X_train.shape[0]} |\n"
        + f" Test samples: {X_test.shape[0]} |\n"
    )
    print("=" * 80)

    logger.info(f"Training KNN Classifier; K (neighbors): {k}...")

    model = KNNClassifier(k=k, distance_metric="euclidean")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    print("=" * 80)
    print("Test Set Results:")
    print(
        "Metrics- \n"
        + f" Accuracy: {accuracy:.4f} |\n"
    )

    if verbose:
        print("=" * 80)
        print("Verbose Output:")
        print("First 5 test samples:")
        for i in range(min(5, len(y_test))):
            print(f"Sample {i+1}: Actual={y_test[i]}, Predicted={y_pred[i]}")

    print("=" * 80)
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
