"""Module for cross-validation."""

from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm


def calculate_best_parameter_using_cross_validation(
    X, Y, lambdas, B_estimator, regret_function, k_folds, total_iterations
):
    """
    Perform k-fold cross-validation to find the best lambda value.

    Parameters:
    -----------
    X : array-like, shape (T, n, d)
        The context data.
    Y : array-like, shape (T, n)
        The reward data.
    - lambdas:
        List of lambda values to try.
    - B_estimator:
        Function that estimates the matrix B and that has lambda as a parameter.
    - regret_function:
        Function that calculates the regret list iterations.
    - k_folds:
        Number of folds to use in cross-validation.
    - total_iterations:
        Number of iterations to run the regret calculation

    Returns:
    --------
    - best_lambda:
        The best lambda value found.
    """
    kf = KFold(n_splits=k_folds)
    best_lambda = None
    best_score = float("inf")

    for lambda_val in lambdas:
        fold_scores = []

        for train_index, val_index in tqdm(
            kf.split(X), desc="Folds", leave=False, total=k_folds
        ):
            X_train, X_val = X[train_index], X[val_index]
            Y_train, Y_val = Y[train_index], Y[val_index]

            B_estimated = B_estimator(X_train, Y_train, lambda_val)

            regret_scores = regret_function(X_val, Y_val, B_estimated, total_iterations)
            score = np.mean(regret_scores)
            fold_scores.append(score)

        avg_score = np.mean(fold_scores)

        if avg_score < best_score:
            best_score = avg_score
            best_lambda = lambda_val

    return best_lambda


def estimate_B_using_cross_validation(
    X, Y, lambdas, B_estimator, regret_function, k_folds=5, total_iterations=20
):
    """
    Estimate the matrix B using cross-validation to find the best lambda value.

    Parameters:
    -----------
    X : array-like, shape (T, n, d)
        The context data.
    Y : array-like, shape (T, n)
        The reward data.
    - lambdas:
        List of lambda values to try.
    - B_estimator:
        Function that estimates the matrix B and that has lambda as a parameter.
    - regret_function:
        Function that calculates the regret list iterations.
    - k_folds:
        Number of folds to use in cross-validation.
    - total_iterations:
        Number of iterations to run the regret calculation.

    Returns:
    --------
    - B:
        The estimated matrix B.
    """

    best_lambda = calculate_best_parameter_using_cross_validation(
        X, Y, lambdas, B_estimator, regret_function, k_folds, total_iterations
    )
    B = B_estimator(X, Y, best_lambda)
    return B
