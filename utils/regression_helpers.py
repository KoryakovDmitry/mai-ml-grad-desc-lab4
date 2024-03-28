import copy

from modules.linear_regression import LinearRegression


def r2_score(y_true, y_pred):
    """
    Calculate the R-squared score.

    Parameters:
    - y_true: array-like of shape (n_samples,) or (n_samples, n_outputs)
              Ground truth (correct) target values.
    - y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs)
              Estimated target values.

    Returns:
    - score : float
              R^2 (coefficient of determination) regression score function.
    """
    # Calculate the mean of true values
    y_true_mean = sum(y_true) / len(y_true)

    # Calculate the total sum of squares (TSS)
    tss = sum((y - y_true_mean) ** 2 for y in y_true)

    # Calculate the residual sum of squares (RSS)
    rss = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))

    # Calculate the R^2 score
    r2 = 1 - rss / tss

    return r2


# Worker function
def fit_and_evaluate(cfg_name, lambda_value, descent_config, tolerance, max_iter, x, y, X_train, y_train, X_val, y_val):
    descent_config = copy.deepcopy(descent_config)
    descent_config['kwargs']['lambda_'] = lambda_value

    regression = LinearRegression(
        descent_config=descent_config,
        tolerance=tolerance,
        max_iter=max_iter
    )
    regression.fit(x, y)

    train_error = regression.calc_loss(X_train, y_train)
    train_r2 = r2_score(y_true=y_train, y_pred=regression.predict(X_train))
    val_error = regression.calc_loss(X_val, y_val)
    val_r2 = r2_score(y_true=y_val, y_pred=regression.predict(X_val))

    result = {
        'lambda': lambda_value,
        'train_error': train_error,
        'train_r2': train_r2,
        'val_error': val_error,
        'val_r2': val_r2,
        'last_iteration_step': regression.last_iter,
    }

    return cfg_name, result
