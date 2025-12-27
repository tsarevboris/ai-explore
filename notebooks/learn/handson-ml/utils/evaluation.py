"""Model evaluation and hyperparameter search utilities."""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import root_mean_squared_error


def fit_evaluate(model, X, y, cv=10, name=None):
    """Fit model and evaluate with training and cross-validation metrics."""

    if name:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")

    print("  Training...", end='', flush=True)
    start_time = time.perf_counter()
    model.fit(X, y)
    fit_time = time.perf_counter() - start_time
    print(f"\r  Training...   ✓ ({fit_time:.2f}s)")

    predictions = model.predict(X)
    rmse = root_mean_squared_error(y, predictions)

    print(f"    RMSE:       {rmse:>12,.2f}")

    if cv is not None and cv > 0:
        print(f"  Cross-validation (k={cv})...", end='', flush=True)
        cv_start = time.perf_counter()
        scores = -cross_val_score(
            model, X, y,
            scoring="neg_root_mean_squared_error",
            cv=cv
        )
        cv_time = time.perf_counter() - cv_start
        print(f"\r  Cross-validation (k={cv})... ✓ ({cv_time:.2f}s)")

        cv_stats = pd.Series(scores).describe()
        print(f"    Mean:       {cv_stats['mean']:>12,.2f}")
        print(f"    Std:        {cv_stats['std']:>12,.2f}")


def randomized_search(model, param_distributions, X, y,
                      n_iter=10, cv=3, scoring="neg_root_mean_squared_error",
                      random_state=42, name=None):
    """Perform randomized hyperparameter search and print results."""

    if name:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")

    print(f"  Searching {n_iter} parameter combinations...", end='', flush=True)
    start_time = time.perf_counter()

    search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        verbose=0,
        return_train_score=True
    )

    search.fit(X, y)
    search_time = time.perf_counter() - start_time
    print(f"\r  Searching {n_iter} parameter combinations... ✓ ({search_time:.2f}s)")

    # Best results
    print(f"\n  Best Score:     {-search.best_score_:>12,.2f}")
    print(f"  Best Parameters:")
    for param, value in search.best_params_.items():
        param_short = param.split('__')[-1]
        if isinstance(value, (int, np.integer)):
            print(f"    {param_short:20s} = {value}")
        elif isinstance(value, (float, np.floating)):
            print(f"    {param_short:20s} = {value:.6f}")
        else:
            print(f"    {param_short:20s} = {value}")

    return search


def grid_search(model, param_grid, X, y,
                cv=3, scoring="neg_root_mean_squared_error", name=None):
    """Perform grid search and print results."""

    if name:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")

    # Count total combinations
    n_combinations = np.prod([len(v) if isinstance(v, list) else 1
                              for v in param_grid.values()])

    print(f"  Searching {n_combinations} parameter combinations...", end='', flush=True)
    start_time = time.perf_counter()

    search = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        verbose=0,
        return_train_score=True
    )

    search.fit(X, y)
    search_time = time.perf_counter() - start_time
    print(f"\r  Searching {n_combinations} parameter combinations... ✓ ({search_time:.2f}s)")

    # Best results
    print(f"\n  Best Score:     {-search.best_score_:>12,.2f}")
    print(f"  Best Parameters:")
    for param, value in search.best_params_.items():
        param_short = param.split('__')[-1]
        if isinstance(value, (int, np.integer)):
            print(f"    {param_short:20s} = {value}")
        elif isinstance(value, (float, np.floating)):
            print(f"    {param_short:20s} = {value:.6f}")
        else:
            print(f"    {param_short:20s} = {value}")

    return search
