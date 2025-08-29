import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import seaborn as sns
from typing import TypeAlias
from typing import Optional, Any
Number: TypeAlias = int | float

# --PREPARE THE FEATURES & TARGET SETS--

def get_features_targets(df: pd.DataFrame, 
                         feature_names: list[str], 
                         target_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_feature: pd.DataFrame = df[feature_names]
    df_target: pd.DataFrame = df[target_names]
    return df_feature, df_target


def split_data(df_feature: pd.DataFrame, 
               df_target: pd.DataFrame, 
               random_state: Optional[int]=None, 
               test_size: float=0.5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indexes: pd.Index = df_feature.index
    if random_state != None:
        np.random.seed(random_state)
    k: int = int(test_size * len(indexes))
    test_index: pd.Index = np.random.choice(indexes, k, replace=False)
    train_index: pd.Index = indexes.drop(test_index)
    df_feature_train: pd.DataFrame = df_feature.loc[train_index, :]
    df_feature_test: pd.DataFrame = df_feature.loc[test_index, :]
    df_target_train: pd.DataFrame = df_target.loc[train_index, :]
    df_target_test: pd.DataFrame = df_target.loc[test_index, :]
    return df_feature_train, df_feature_test, df_target_train, df_target_test


def normalize_z(array: np.ndarray, columns_means: Optional[np.ndarray]=None, 
                columns_stds: Optional[np.ndarray]=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert columns_means is None or columns_means.shape == (1, array.shape[1])
    assert columns_stds is None or columns_stds.shape == (1, array.shape[1])
    if columns_means is None:
        columns_means = np.mean(array, axis=0).reshape(1, -1)
    if columns_stds is None:
        columns_stds = np.std(array, axis=0, ddof=0).reshape(1, -1)
    out = (array - columns_means) / columns_stds
    assert out.shape == array.shape
    assert columns_means.shape == (1, array.shape[1])
    assert columns_stds.shape == (1, array.shape[1])
    return out, columns_means, columns_stds


def prepare_feature(np_feature: np.ndarray) -> np.ndarray:
    rows, cols = np_feature.shape
    np_ones = np.ones((rows, 1))
    X:np.ndarray = np.concatenate((np_ones, np_feature), axis = 1 )
    return X

# --BUILD & TEST LINEAR REGRESSION MODEL--

def calc_linreg(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    result = np.matmul(X, beta)
    assert result.shape == (X.shape[0], 1)
    return result


def compute_cost_linreg(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    m: int = X.shape[0]
    y_pred = np.matmul(X, beta)
    error = y_pred - y 
    error_sq = np.matmul(error.T, error)
    J: np.ndarray = (1/(2*m)) * error_sq
    assert J.shape == (1, 1)
    return np.squeeze(J)


def gradient_descent_linreg(X: np.ndarray, 
                            y: np.ndarray, 
                            beta: np.ndarray, 
                            alpha: float, 
                            num_iters: int) -> tuple[np.ndarray, np.ndarray]:
    m = X.shape[0]
    J_storage = np.zeros((num_iters, 1))
    for i in range(num_iters):
        y_hat = calc_linreg(X, beta)
        error = y_hat - y
        dJ_db = np.matmul(X.transpose(), error)
        beta = beta - (alpha / m) * dJ_db
        J_storage[i, 0] = compute_cost_linreg(X, y, beta)
    assert beta.shape == (X.shape[1], 1)
    assert J_storage.shape == (num_iters, 1)
    return beta, J_storage


def predict_linreg(df_feature: pd.DataFrame, 
                   beta: np.ndarray, 
                   means: Optional[np.ndarray]=None, 
                   stds: Optional[np.ndarray]=None) -> np.ndarray:
    normalized_feature, means, stds = normalize_z(df_feature.values, means, stds)
    X = prepare_feature(normalized_feature)
    result = calc_linreg(X, beta)
    return result


def build_model_linreg(df_feature_train: pd.DataFrame,
                       df_target_train: pd.DataFrame,
                       beta: Optional[np.ndarray] = None,
                       alpha: float = 0.01,
                       iterations: int = 1500) -> tuple[dict[str, Any], np.ndarray]:
    if beta is None:
        beta = np.zeros((df_feature_train.shape[1] + 1, 1)) 
    assert beta.shape == (df_feature_train.shape[1] + 1, 1)
    model: dict[str, Any] = {}
    array_feature_train_z, means_train, stds_train = normalize_z(df_feature_train.to_numpy())
    X = prepare_feature(array_feature_train_z)
    y = df_target_train.to_numpy()
    beta, J_storage = gradient_descent_linreg(X, y, beta, alpha, iterations)
    model['beta'] = beta
    model['means'] = means_train
    model['stds'] = stds_train
    assert model["beta"].shape == (df_feature_train.shape[1] + 1, 1)
    assert model["means"].shape == (1, df_feature_train.shape[1])
    assert model["stds"].shape == (1, df_feature_train.shape[1])
    assert J_storage.shape == (iterations, 1)
    return model, J_storage

# --EVALUATE LINEAR REGRESSION MODEL--

def r2_score(y: np.ndarray, ypred: np.ndarray) -> float:
    error = y - ypred
    SSres = np.matmul(error.transpose(), error)
    y_mean = np.mean(y)
    error_mean = y - y_mean
    SStot = np.matmul(error_mean.transpose(), error_mean)
    r2 = 1 - SSres/SStot
    return np.squeeze(r2)


def mean_squared_error(target: np.ndarray, pred: np.ndarray) -> float:
    error = target - pred
    sum_of_squares = np.matmul(error.transpose(), error)
    n = target.shape[0]
    return np.squeeze(sum_of_squares / n)
