"""
02_classification_pipeline.py
K-Means spatial regimes (k=5), retrospective sliding window (W=6),
Stratified 5-Fold Cross-Validation, LSTM, Bi-LSTM, XGBoost, CatBoost,
and Majority Voting Ensemble Evaluation.
"""

import os
import random
import time
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Bidirectional,
    Dense,
    Dropout,
    Input,
    Masking,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import xgboost as xgb

# ==============================================================================
# CONFIGURATION & REPRODUCIBILITY SEEDS
# ==============================================================================
RANDOM_STATE = 42
WINDOW_SIZE = 6

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "merged_dataset_with_diffuse_attenuation_coefficient.csv")

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def mean_confidence_interval(data, confidence: float = 0.95):
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return mean, h


def build_standard_lstm(nfeatures: int, n_classes: int):
    """Standard LSTM architecture matching Table 1 (64 units, Dense 32)."""
    model = Sequential([
        Input(shape=(WINDOW_SIZE, nfeatures)),
        Masking(mask_value=0.0),
        LSTM(64, activation="tanh"),
        Dropout(0.30),
        Dense(32, activation="relu"),
        Dense(n_classes, activation="softmax")
    ])
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"]
    )
    return model


def build_bilstm_model(nfeatures: int, n_classes: int):
    """Bidirectional LSTM matching Table 1 (L1=192, L2=96, Dense 64)."""
    model = Sequential([
        Input(shape=(WINDOW_SIZE, nfeatures)),
        Masking(mask_value=0.0),
        Bidirectional(LSTM(192, activation="tanh", return_sequences=True)),
        BatchNormalization(),
        Dropout(0.25),
        Bidirectional(LSTM(96, activation="tanh")),
        BatchNormalization(),
        Dropout(0.25),
        Dense(64, activation="relu"),
        Dense(n_classes, activation="softmax")
    ])
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"]
    )
    return model


def main():
    start_time = time.time()
    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["month"] = df["date"].dt.month
    df["season"] = (df["month"] % 12 + 3) // 3
    df["lat_sin"] = np.sin(np.radians(df["latitude"]))
    df["lat_cos"] = np.cos(np.radians(df["latitude"]))
    df["lon_sin"] = np.sin(np.radians(df["longitude"]))
    df["lon_cos"] = np.cos(np.radians(df["longitude"]))

    coords = df[["latitude", "longitude"]].dropna()
    kmeans = KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init="auto")
    df["region"] = kmeans.fit_predict(coords)

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

    target_text_col = "microplastic concentration class text"
    df = df.dropna(subset=all_features + [target_text_col])

    categorical_cols = ["marine setting", "month", "season", "region"]
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    le_target = LabelEncoder()
    df["target"] = le_target.fit_transform(df[target_text_col].astype(str))
    n_classes = len(le_target.classes_)

    sequences, targets = [], []
    df_sorted = df.sort_values(["latitude", "longitude", "date"])
    for _, group in df_sorted.groupby(["latitude", "longitude"]):
        X_group = group[all_features].values
        y_group = group["target"].values
        if len(X_group) < WINDOW_SIZE:
            continue
        for i in range(len(X_group) - WINDOW_SIZE + 1):
            sequences.append(X_group[i : i + WINDOW_SIZE])
            targets.append(y_group[i + WINDOW_SIZE - 1])

    X_seq = np.array(sequences)
    y_seq = np.array(targets)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    metrics_dict = {
        "BiLSTM": [],
        "LSTM": [],
        "XGBoost": [],
        "CatBoost": [],
        "Ensemble": [],
    }

    fold = 1
    for train_idx, test_idx in kf.split(X_seq, y_seq):
        print(f"\n=== Fold {fold} ===")
        X_train_seq, X_test_seq = X_seq[train_idx], X_seq[test_idx]
        y_train_seq_fold, y_test_seq_fold = y_seq[train_idx], y_seq[test_idx]

        y_train_cat = to_categorical(y_train_seq_fold, num_classes=n_classes)

        nsamples, ntimesteps, nfeatures = X_train_seq.shape
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(
            X_train_seq.reshape(-1, nfeatures)
        ).reshape(X_train_seq.shape)
        X_test_scaled = scaler.transform(
            X_test_seq.reshape(-1, nfeatures)
        ).reshape(X_test_seq.shape)

        # 1. Standard LSTM
        lstm_model = build_standard_lstm(nfeatures, n_classes)
        lstm_model.fit(X_train_scaled, y_train_cat, epochs=50, batch_size=64, verbose=0)
        lstm_pred = np.argmax(lstm_model.predict(X_test_scaled, verbose=0), axis=1)
        metrics_dict["LSTM"].append(lstm_pred)

        # 2. Bidirectional LSTM
        bilstm_model = build_bilstm_model(nfeatures, n_classes)
        bilstm_model.fit(X_train_scaled, y_train_cat, epochs=50, batch_size=64, verbose=0)
        bilstm_pred = np.argmax(bilstm_model.predict(X_test_scaled, verbose=0), axis=1)
        metrics_dict["BiLSTM"].append(bilstm_pred)

        # 3. XGBoost Classifier
        X_train_flat = X_train_seq[:, -1, :]
        X_test_flat = X_test_seq[:, -1, :]
        classes_arr = np.unique(y_train_seq_fold)
        weights = class_weight.compute_class_weight(
            class_weight="balanced", classes=classes_arr, y=y_train_seq_fold
        )
        sample_weights = pd.Series(y_train_seq_fold).map(dict(zip(classes_arr, weights)))

        xgb_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=0.5,
            random_state=RANDOM_STATE
        )
        xgb_model.fit(X_train_flat, y_train_seq_fold, sample_weight=sample_weights)
        xgb_pred = xgb_model.predict(X_test_flat)
        metrics_dict["XGBoost"].append(xgb_pred)

        # 4. CatBoost Classifier
        cat_model = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=4,
            loss_function="MultiClass",
            verbose=0,
            random_seed=RANDOM_STATE,
            class_weights=dict(zip(classes_arr, weights))
        )
        X_train_cat = pd.DataFrame(X_train_flat, columns=all_features)
        X_test_cat = pd.DataFrame(X_test_flat, columns=all_features)
        for col in categorical_cols:
            X_train_cat[col] = X_train_cat[col].astype(str)
            X_test_cat[col] = X_test_cat[col].astype(str)
        cat_model.fit(X_train_cat, y_train_seq_fold)
        cat_pred = cat_model.predict(X_test_cat).flatten().astype(int)
        metrics_dict["CatBoost"].append(cat_pred)

        # 5. Majority Voting Ensemble
        ensemble_stack = np.vstack([bilstm_pred, lstm_pred, xgb_pred, cat_pred])
        ensemble_mode, _ = stats.mode(ensemble_stack, axis=0, keepdims=False)
        ensemble_final = ensemble_mode.flatten()
        metrics_dict["Ensemble"].append(ensemble_final)

        fold += 1

    for model_name, preds_list in metrics_dict.items():
        (
            accs,
            f1_macros,
            f1_weighted,
            prec_macros,
            prec_weighted,
            rec_macros,
            rec_weighted,
        ) = ([], [], [], [], [], [], [])
        for i, y_pred in enumerate(preds_list):
            _, test_idx = list(kf.split(X_seq, y_seq))[i]
            y_true = y_seq[test_idx]
            accs.append(accuracy_score(y_true, y_pred))
            f1_macros.append(f1_score(y_true, y_pred, average="macro", zero_division=0))
            f1_weighted.append(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            prec_macros.append(precision_score(y_true, y_pred, average="macro", zero_division=0))
            prec_weighted.append(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            rec_macros.append(recall_score(y_true, y_pred, average="macro", zero_division=0))
            rec_weighted.append(recall_score(y_true, y_pred, average="weighted", zero_division=0))

        print(f"\n--- {model_name} 5-Fold Metrics (95% CI) ---")
        for name, values in zip(
            [
                "Accuracy",
                "F1 Macro",
                "F1 Weighted",
                "Precision Macro",
                "Precision Weighted",
                "Recall Macro",
                "Recall Weighted",
            ],
            [
                accs,
                f1_macros,
                f1_weighted,
                prec_macros,
                prec_weighted,
                rec_macros,
                rec_weighted,
            ],
        ):
            mean, h = mean_confidence_interval(values)
            print(f"{name:<20}: {mean:.4f} ± {h:.4f}")

    elapsed = time.time() - start_time
    print(f"\nElapsed time: {int(elapsed//60)} min {int(elapsed%60)} sec")


if __name__ == "__main__":
    main()
