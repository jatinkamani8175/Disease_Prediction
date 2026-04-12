"""
model/train_model.py — Standalone script to train and save the ML model.

Usage:
    python model/train_model.py

What it does:
  1. Loads the symptoms-to-disease CSV (data/symptoms_disease.csv)
  2. Stratified-samples up to `SAMPLES_PER_DISEASE` rows per disease
     to keep the model file GitHub-friendly (< 20 MB)
  3. Trains a RandomForestClassifier
  4. Evaluates on a hold-out test split
  5. Saves the model to model/model.pkl  (compressed)
  6. Saves the ordered feature column list to model/feature_columns.csv

Dataset format expected (data/symptoms_disease.csv):
    diseases,symptom_1,symptom_2,...,symptom_n
    acne,0,1,0,...,0
    pneumonia,1,0,1,...,1
    ...
"""

import os
import sys
import time
import gc

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── Configuration ─────────────────────────────────────────────────────────────
SAMPLES_PER_DISEASE = 40          # max rows kept per disease (keeps model small)
N_ESTIMATORS        = 150         # number of trees
MAX_DEPTH           = 25          # tree depth limit
TEST_SIZE           = 0.20        # fraction used for evaluation
RANDOM_STATE        = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SCRIPT_DIR)

DATA_PATH    = os.path.join(_ROOT_DIR, "data", "symptoms_disease.csv")
MODEL_PATH   = os.path.join(_SCRIPT_DIR, "model.pkl")
COLUMNS_PATH = os.path.join(_SCRIPT_DIR, "feature_columns.csv")


def load_and_sample(path: str, samples_per_disease: int) -> pd.DataFrame:
    """Load the CSV and stratified-sample to reduce memory footprint."""
    print(f"[1/5] Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"      Full dataset  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"      Unique diseases: {df['diseases'].nunique()}")

    print(f"[2/5] Sampling up to {samples_per_disease} rows per disease …")
    target_col = df.columns[0]   # 'diseases'

    sampled_idx = []
    for _, grp in df.groupby(target_col):
        n = min(len(grp), samples_per_disease)
        sampled_idx.extend(grp.sample(n, random_state=RANDOM_STATE).index.tolist())

    df_s = df.loc[sampled_idx].reset_index(drop=True)
    print(f"      Sampled dataset: {df_s.shape[0]:,} rows")
    del df
    gc.collect()
    return df_s


def train(df: pd.DataFrame) -> tuple:
    """
    Split data, train RandomForest, return (model, X_test, y_test, feature_cols).
    """
    target_col   = "diseases"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].astype(np.int8)
    y = df[target_col]

    print(f"[3/5] Splitting data — test size: {TEST_SIZE:.0%} …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    del X
    gc.collect()

    print(f"[4/5] Training RandomForestClassifier "
          f"(n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}) …")
    t0  = time.time()
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"      Training completed in {elapsed:.1f} s")

    return clf, X_test, y_test, feature_cols


def evaluate(clf, X_test, y_test) -> None:
    """Print accuracy and a short classification report."""
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'─'*50}")
    print(f"  Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"{'─'*50}\n")

    # Show per-class report for a random subset of diseases to keep output short
    sample_diseases = sorted(y_test.unique())[:15]
    mask = y_test.isin(sample_diseases)
    if mask.sum() > 0:
        print("Classification report (first 15 disease classes shown):")
        print(
            classification_report(
                y_test[mask],
                y_pred[mask],
                target_names=sample_diseases,
                zero_division=0,
            )
        )


def save_artifacts(clf, feature_cols: list[str]) -> None:
    """Persist model and feature column list to disk."""
    print("[5/5] Saving artifacts …")

    joblib.dump(clf, MODEL_PATH, compress=3)
    size_mb = os.path.getsize(MODEL_PATH) / (1024 ** 2)
    print(f"      model.pkl saved  : {MODEL_PATH}  ({size_mb:.1f} MB)")

    pd.Series(feature_cols).to_csv(COLUMNS_PATH, index=False, header=False)
    print(f"      feature_columns.csv saved: {COLUMNS_PATH}")


def main() -> None:
    print("=" * 60)
    print("  Disease Prediction — Model Training Script")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"\n[ERROR] Dataset not found at:\n  {DATA_PATH}")
        print("Place 'symptoms_disease.csv' in the data/ folder and re-run.")
        sys.exit(1)

    df               = load_and_sample(DATA_PATH, SAMPLES_PER_DISEASE)
    clf, X_test, y_test, feature_cols = train(df)
    evaluate(clf, X_test, y_test)
    save_artifacts(clf, feature_cols)

    print("\n✅  Training complete! You can now run the Streamlit app:")
    print("    streamlit run app.py\n")


if __name__ == "__main__":
    main()
