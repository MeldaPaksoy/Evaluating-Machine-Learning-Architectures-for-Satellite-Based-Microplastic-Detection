"""
03_regression_pipeline.py
Continuous concentration regression on native physical scale (pieces/m³):
1. RandomizedSearchCV Hyperparameter Optimization (XGBoost & Random Forest)
2. Soft Voting (MAE-weighted) and Ridge Meta-Stacking
3. Scikit-Learn StackingRegressor Pipeline (Out-Of-Fold Safe)
4. 5-Fold Cross-Validation (Linear Regression with log1p & Random Forest max_depth=6)
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

RANDOM_STATE = 42
N_ITER_SEARCH = 30
CV_FOLDS = 3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "merged_dataset_with_diffuse_attenuation_coefficient.csv")


def calculate_regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def print_regression_row(model_name: str, mae: float, rmse: float, r2: float) -> str:
    return f"| {model_name.ljust(25)} | {mae:.4f} | {rmse:.4f} | {r2:.4f} |"


def main():
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["month"] = df["date"].dt.month
    df["season"] = (df["month"] % 12 + 3) // 3

    coords = df[["latitude", "longitude"]].dropna()
    kmeans = KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=10)
    df["region"] = kmeans.fit_predict(coords).astype(int)

    all_features = [
        "latitude",
        "longitude",
        "marine setting",
        "sea surface temperature",
        "chlor_a",
        "Kd_490_mean",
        "month",
        "season",
        "region",
    ]
    target = "microplastics measurement"
    categorical_cols = ["marine setting", "month", "season", "region"]

    df_clean = df.dropna(subset=all_features + [target]).copy()

    for col in categorical_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))

    X = df_clean[all_features].values
    y = df_clean[target].values

    # Train %70, Val %10, Test %20 (80% Development, 20% Quarantined Test)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.125, random_state=RANDOM_STATE
    )
    y_true = y_test

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 1. Base Modeller ve Hiperparametre Optimizasyonu
    print("\n--- Training Linear Regression ---")
    lr_model = LinearRegression(fit_intercept=True)
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_test_scaled)
    mae_lr_test, rmse_lr_test, r2_lr_test = calculate_regression_metrics(y_true, lr_preds)

    print("\n--- Hyperparameter Tuning XGBoost ---")
    xgb_param_dist = {
        "n_estimators": randint(200, 1000),
        "learning_rate": uniform(0.01, 0.1),
        "max_depth": randint(3, 10),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
    }
    xgb_search = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            eval_metric="mae",
        ),
        param_distributions=xgb_param_dist,
        n_iter=N_ITER_SEARCH,
        scoring=mae_scorer,
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    xgb_search.fit(X_train_full, y_train_full)
    xgb_model = xgb_search.best_estimator_
    xgb_preds = xgb_model.predict(X_test)
    mae_xgb_test, rmse_xgb_test, r2_xgb_test = calculate_regression_metrics(y_true, xgb_preds)
    print(f"XGBoost Best Params: {xgb_search.best_params_}")

    print("\n--- Hyperparameter Tuning RandomForest ---")
    rf_param_dist = {
        "n_estimators": randint(100, 800),
        "max_depth": [6],
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": uniform(0.5, 0.5),
    }
    rf_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=RANDOM_STATE, max_depth=6, n_jobs=-1),
        param_distributions=rf_param_dist,
        n_iter=N_ITER_SEARCH,
        scoring=mae_scorer,
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    rf_search.fit(X_train_full, y_train_full)
    rf_model = rf_search.best_estimator_
    rf_preds = rf_model.predict(X_test)
    mae_rf_test, rmse_rf_test, r2_rf_test = calculate_regression_metrics(y_true, rf_preds)
    print(f"RandomForest Best Params: {rf_search.best_params_}")

    # 2. Ensemble: Hard Voting, Soft Voting ve Stacking
    ensemble_preds_matrix = np.vstack([lr_preds, xgb_preds, rf_preds])
    ensemble_y_pred_hard = np.mean(ensemble_preds_matrix, axis=0)
    mae_hard, rmse_hard, r2_hard = calculate_regression_metrics(y_true, ensemble_y_pred_hard)

    print("\n--- Soft Voting ---")
    lr_val_preds = lr_model.predict(X_val_scaled)
    mae_lr_val = mean_absolute_error(y_val, lr_val_preds)
    mae_xgb_val = -xgb_search.best_score_
    mae_rf_val = -rf_search.best_score_

    maes = np.array([mae_lr_val, mae_xgb_val, mae_rf_val])
    weights_raw = 1 / maes
    weights = weights_raw / np.sum(weights_raw)

    ensemble_y_pred_soft = np.average(
        ensemble_preds_matrix, axis=0, weights=weights
    )
    mae_soft, rmse_soft, r2_soft = calculate_regression_metrics(y_true, ensemble_y_pred_soft)

    print("\n--- Stacking (Ridge Meta-Model) ---")
    meta_features_val = np.column_stack(
        (lr_val_preds, xgb_search.predict(X_val), rf_search.predict(X_val))
    )
    meta_model = Ridge(alpha=1.0, fit_intercept=True)
    meta_model.fit(meta_features_val, y_val)
    meta_features_test = np.column_stack((lr_preds, xgb_preds, rf_preds))
    ensemble_y_pred_stacking = meta_model.predict(meta_features_test)
    mae_stacking, rmse_stacking, r2_stacking = calculate_regression_metrics(y_true, ensemble_y_pred_stacking)

    print("\n" + "=" * 70)
    print("Scores")
    print("=" * 70)
    header = "| Model                     | MAE (Test) | RMSE (Test) | R² (Test) |"
    print(header)
    print("|" + "-" * 27 + "|" + "-" * 12 + "|" + "-" * 13 + "|" + "-" * 11 + "|")
    print(print_regression_row("Linear Regression", mae_lr_test, rmse_lr_test, r2_lr_test))
    print(print_regression_row("XGBoost", mae_xgb_test, rmse_xgb_test, r2_xgb_test))
    print(print_regression_row("RandomForest", mae_rf_test, rmse_rf_test, r2_rf_test))
    print("|" + "=" * 27 + "|" + "=" * 12 + "|" + "=" * 13 + "|" + "=" * 11 + "|")
    print(print_regression_row("Hard Voting (Simple Avg)", mae_hard, rmse_hard, r2_hard))
    print(print_regression_row("Soft Voting (Weighted Avg)", mae_soft, rmse_soft, r2_soft))
    print(print_regression_row("Stacking (Ridge Meta)", mae_stacking, rmse_stacking, r2_stacking))
    print("=" * 70)

    # 3. Sklearn StackingRegressor (OOF Safe)
    print("\n--- Training Sklearn StackingRegressor (OOF Safe) ---")
    stack_pipeline = StackingRegressor(
        estimators=[("xgb", xgb_model), ("rf", rf_model)],
        final_estimator=Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, fit_intercept=True))]
        ),
        cv=5,
        n_jobs=-1
    )
    stack_pipeline.fit(X_train, y_train)
    stack_preds_pipe = stack_pipeline.predict(X_test)
    mae_sp, rmse_sp, r2_sp = calculate_regression_metrics(y_true, stack_preds_pipe)
    print(f"StackingRegressor OOF: MAE={mae_sp:.4f}, RMSE={rmse_sp:.4f}, R2={r2_sp:.4f}")

    # 4. 5-Fold Cross-Validation (Log Dönüşümlü Linear Regression - Snapshot Base)
    print("\n--- Linear Regression 5-Fold CV (np.log1p) ---")
    y_train_log = np.log1p(y_train_full)
    kf_reg = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    mae_log_scores, rmse_log_scores, r2_log_scores = [], [], []

    for tr_i, val_i in kf_reg.split(X_train_full):
        X_tr_fold, X_v_fold = X_train_full[tr_i], X_train_full[val_i]
        y_tr_fold, y_v_fold = y_train_log[tr_i], y_train_log[val_i]

        sc_fold = StandardScaler()
        X_tr_fold_s = sc_fold.fit_transform(X_tr_fold)
        X_v_fold_s = sc_fold.transform(X_v_fold)

        lr_log = LinearRegression(fit_intercept=True)
        lr_log.fit(X_tr_fold_s, y_tr_fold)

        y_v_pred_log = lr_log.predict(X_v_fold_s)
        y_v_pred_orig = np.expm1(y_v_pred_log)
        y_v_true_orig = np.expm1(y_v_fold)

        mae_l, rmse_l, r2_l = calculate_regression_metrics(y_v_true_orig, y_v_pred_orig)
        mae_log_scores.append(mae_l)
        rmse_log_scores.append(rmse_l)
        r2_log_scores.append(r2_l)

    print(f"5-Fold CV Mean MAE (LR log1p): {np.mean(mae_log_scores):.4f}")
    print(f"5-Fold CV Mean RMSE (LR log1p): {np.mean(rmse_log_scores):.4f}")
    print(f"5-Fold CV Mean R2 (LR log1p): {np.mean(r2_log_scores):.4f}")

    # 5. 5-Fold Cross-Validation (Random Forest, max_depth=6)
    print("\n--- Random Forest 5-Fold CV ---")
    kf_rf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rf_mae_scores, rf_rmse_scores, rf_r2_scores = [], [], []

    for tr_idx, val_idx in kf_rf.split(X_train_full):
        X_tr_k, X_val_k = X_train_full[tr_idx], X_train_full[val_idx]
        y_tr_k, y_val_k = y_train_full[tr_idx], y_train_full[val_idx]

        sc_rf = StandardScaler()
        X_tr_k_s = sc_rf.fit_transform(X_tr_k)
        X_val_k_s = sc_rf.transform(X_val_k)

        rf_cv_m = RandomForestRegressor(
            n_estimators=500, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1
        )
        rf_cv_m.fit(X_tr_k_s, y_tr_k)
        y_pred_rf = rf_cv_m.predict(X_val_k_s)

        m_rf, rm_rf, r_rf = calculate_regression_metrics(y_val_k, y_pred_rf)
        rf_mae_scores.append(m_rf)
        rf_rmse_scores.append(rm_rf)
        rf_r2_scores.append(r_rf)

    print(f"5-Fold CV Mean MAE (Random Forest): {np.mean(rf_mae_scores):.4f}")
    print(f"5-Fold CV Mean RMSE (Random Forest): {np.mean(rf_rmse_scores):.4f}")
    print(f"5-Fold CV Mean R2 (Random Forest): {np.mean(rf_r2_scores):.4f}")


if __name__ == "__main__":
    main()
