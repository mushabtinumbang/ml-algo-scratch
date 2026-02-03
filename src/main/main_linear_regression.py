import numpy as np
import pandas as pd
import click
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.subplots as sp
from src.features.linear_regression import LinearRegression
from src.utilities.config_ import img_path, log_path
from src.utilities.utils import setup_logger
from loguru import logger


# log
setup_logger("linear_regression", log_path)

@click.command()
@click.option('--n_samples', type=int, default=1000, help='Number of samples')
@click.option('--n_features', type=int, default=1, help='Number of features')
@click.option('--noise', type=float, default=20.0, help='Noise level')
@click.option('--random_state', type=int, default=42, help='Random state for reproducibility')
@click.option('--test_size', type=float, default=0.2, help='Test set size')
@click.option('--learning_rate', type=float, default=0.01, help='Learning rate')
@click.option('--n_iters', type=int, default=1000, help='Number of iterations')
@click.option('--optimization_metric', type=click.Choice(['mse', 'mae', 'r2'], case_sensitive=False),
              default='mse', help='Optimization metric')
@click.option('--verbose', is_flag=True, help='Print training progress')
def main(n_samples, n_features, noise, random_state, test_size, learning_rate, n_iters, optimization_metric, verbose):
    """
    Linear Regression from scratch with multiple metrics.
    """
    logger.info("Starting to run Linear Regression ...")

    logger.info(
        "Data Generation Params- \n"
        + f" Samples: {n_samples} |\n"
        + f" Features: {n_features} |\n"
        + f" Noise: {noise} |\n"
        + f" Random state: {random_state} |\n"
    )

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
        n_informative=n_features,
        bias=0.0
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        "Split Results- \n"
        + f" Training samples: {X_train.shape[0]} |\n"
        + f" Test samples: {X_test.shape[0]} |\n"
    )

    logger.info("Training Linear Regression...")
    logger.info(
        "Training Params- \n"
        + f" Optimization metric: {optimization_metric.upper()} |\n"
        + f" Learning rate: {learning_rate} |\n"
        + f" Number of iterations: {n_iters} |\n"
    )

    model = LinearRegression(
        lr=learning_rate,
        n_iters=n_iters,
        fit_intercept=True,
        verbose=verbose,
        optimization_metric=optimization_metric
    )

    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)

    mse = model._compute_mse(y_pred, y_test)
    mae = model._compute_mae(y_pred, y_test)
    r2 = model._compute_r2(y_pred, y_test)

    logger.info("Test Set Results:")
    logger.info(
        "Metrics- \n"
        + f" MSE: {mse:.4f} |\n"
        + f" MAE: {mae:.4f} |\n"
        + f" R2: {r2:.4f} |\n"
    )

    logger.info("Generating visualizations...")

    # Plot Training loss
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=list(range(len(model.loss_history))),
        y=model.loss_history,
        mode='lines',
        name=f'{optimization_metric.upper()} Loss',
        line=dict(color='blue', width=2)
    ))
    fig_loss.update_layout(
        title='Training Loss Over Iterations',
        xaxis_title='Iteration',
        yaxis_title=f'{optimization_metric.upper()} Loss',
        hovermode='x unified',
        template='plotly_white'
    )

    # Plot metrics over iterations
    metrics_df = pd.DataFrame(model.metrics_history)

    fig_metrics = sp.make_subplots(rows=1, cols=3, subplot_titles=('MSE', 'MAE', 'R2'))

    fig_metrics.add_trace(
        go.Scatter(x=list(range(len(metrics_df))), y=metrics_df['mse'], mode='lines', name='MSE'),
        row=1, col=1
    )
    fig_metrics.add_trace(
        go.Scatter(x=list(range(len(metrics_df))), y=metrics_df['mae'], mode='lines', name='MAE'),
        row=1, col=2
    )
    fig_metrics.add_trace(
        go.Scatter(x=list(range(len(metrics_df))), y=metrics_df['r2'], mode='lines', name='R2'),
        row=1, col=3
    )

    fig_metrics.update_layout(
        title='Metrics Over Training Iterations',
        xaxis_title='Iteration',
        hovermode='x unified',
        template='plotly_white'
    )

    # Plot Figure (Only if n_feature == 1)
    if n_features == 1:
        fig_pred = go.Figure()

        # Use first feature as x-axis
        x_axis = X_test[:, 0]

        # Sort by x-axis for clean line plot
        sort_idx = np.argsort(x_axis)
        x_sorted = x_axis[sort_idx]
        y_test_sorted = y_test[sort_idx]
        y_pred_sorted = y_pred[sort_idx]

        fig_pred.add_trace(go.Scatter(
            x=x_sorted,
            y=y_test_sorted,
            mode='markers',
            name='Actual',
            marker=dict(color='blue', size=8, symbol='circle')
        ))

        fig_pred.add_trace(go.Scatter(
            x=x_sorted,
            y=y_pred_sorted,
            mode='lines+markers',
            name='Predicted',
            line=dict(color='red', width=2),
            marker=dict(color='red', size=6, symbol='circle')
        ))

        fig_pred.update_layout(
            title='Actual vs Predicted',
            xaxis_title='Feature Value',
            yaxis_title='Target Value',
            hovermode='x unified',
            template='plotly_white'
        )

    # Save figures
    fig_loss.write_image(str(img_path / 'loss_history.jpg'), format='jpeg')
    fig_metrics.write_image(str(img_path / 'metrics_history.jpg'), format='jpeg')
    logger.info(f"Saved: {img_path / 'loss_history.jpg'}")
    logger.info(f"Saved: {img_path / 'metrics_history.jpg'}")

    if n_features == 1:
        fig_pred.write_image(str(img_path / 'predictions.jpg'), format='jpeg')
        logger.info(f"Saved: {img_path / 'predictions.jpg'}")

    logger.info("Training complete!")

    if model.fit_intercept:
        logger.info(
            "Model Params: \n"
            + f" Coef: {model.coef_} |\n"
            + f" Intercept: {model.intercept_:.4f} |\n"
        )
    else:
        logger.info("Parameters-")
        logger.info(
            "Model Params- \n"
            + f" Coef: {model.coef_} |\n"
        )


if __name__ == '__main__':
    main()
