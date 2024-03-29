import copy
import time

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
def fit_and_evaluate_task51(cfg_name, lambda_value, descent_config, tolerance, max_iter, X_train, y_train, X_val,
                            y_val):
    descent_config = copy.deepcopy(descent_config)
    descent_config['kwargs']['lambda_'] = lambda_value

    regression = LinearRegression(
        descent_config=descent_config,
        tolerance=tolerance,
        max_iter=max_iter
    )
    regression.fit(X_train, y_train)

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


def fit_and_evaluate_task52(cfg_name, lambda_value, descent_config, tolerance, max_iter, X_train, y_train, X_val,
                            y_val, X_test, y_test):
    descent_config = copy.deepcopy(descent_config)
    descent_config['kwargs']['lambda_'] = lambda_value

    regression = LinearRegression(
        descent_config=descent_config,
        tolerance=tolerance,
        max_iter=max_iter
    )
    regression.fit(X_train, y_train)

    train_error = regression.calc_loss(X_train, y_train)
    train_r2 = r2_score(y_true=y_train, y_pred=regression.predict(X_train))
    val_error = regression.calc_loss(X_val, y_val)
    val_r2 = r2_score(y_true=y_val, y_pred=regression.predict(X_val))
    test_error = regression.calc_loss(X_test, y_test)
    test_r2 = r2_score(y_true=y_test, y_pred=regression.predict(X_test))

    result = {
        'lambda': lambda_value,
        'train_error': train_error,
        'train_r2': train_r2,
        'val_error': val_error,
        'val_r2': val_r2,
        'test_error': test_error,
        'test_r2': test_r2,
        'last_iteration_step': regression.last_iter,
        'loss_history': regression.loss_history
    }

    return cfg_name, result


def fit_and_evaluate_task6(cfg_name, batch_size, descent_config, tolerance, max_iter, X_train, y_train, X_val, y_val,
                           X_test, y_test):
    descent_config = copy.deepcopy(descent_config)
    descent_config['kwargs']['batch_size'] = batch_size

    regression = LinearRegression(
        descent_config=descent_config,
        tolerance=tolerance,
        max_iter=max_iter
    )
    start_time = time.time()
    regression.fit(X_train, y_train)
    training_time = time.time() - start_time

    train_error = regression.calc_loss(X_train, y_train)
    train_r2 = r2_score(y_true=y_train, y_pred=regression.predict(X_train))
    val_error = regression.calc_loss(X_val, y_val)
    val_r2 = r2_score(y_true=y_val, y_pred=regression.predict(X_val))
    test_error = regression.calc_loss(X_test, y_test)
    test_r2 = r2_score(y_true=y_test, y_pred=regression.predict(X_test))

    result = {
        'batch_size': int(batch_size),
        'training_time': training_time,
        'train_error': train_error,
        'train_r2': train_r2,
        'val_error': val_error,
        'val_r2': val_r2,
        'test_error': test_error,
        'test_r2': test_r2,
        'last_iteration_step': regression.last_iter,
        'loss_history': regression.loss_history
    }

    return cfg_name, result


def fit_and_evaluate_task7(cfg_name, descent_config, lambda_, mu, tolerance, max_iter, X_train, y_train, X_val, y_val,
                           X_test, y_test):
    descent_config = copy.deepcopy(descent_config)
    descent_config['kwargs']['lambda_'] = lambda_
    if mu != 0:
        descent_config['kwargs']['mu'] = mu

    regression = LinearRegression(
        descent_config=descent_config,
        tolerance=tolerance,
        max_iter=max_iter
    )
    start_time = time.time()
    regression.fit(X_train, y_train)
    training_time = time.time() - start_time

    train_error = regression.calc_loss(X_train, y_train)
    train_r2 = r2_score(y_true=y_train, y_pred=regression.predict(X_train))
    val_error = regression.calc_loss(X_val, y_val)
    val_r2 = r2_score(y_true=y_val, y_pred=regression.predict(X_val))
    test_error = regression.calc_loss(X_test, y_test)
    test_r2 = r2_score(y_true=y_test, y_pred=regression.predict(X_test))

    result = {
        'mu': mu,
        'lambda': lambda_,
        'training_time': training_time,
        'train_error': train_error,
        'train_r2': train_r2,
        'val_error': val_error,
        'val_r2': val_r2,
        'test_error': test_error,
        'test_r2': test_r2,
        'last_iteration_step': regression.last_iter,
        'loss_history': regression.loss_history
    }

    return cfg_name, result
