"""
utils/predictor.py — ML prediction & supporting-information retrieval.

Responsibilities:
  • Load and cache the trained RandomForest model (model.pkl)
  • Predict the most likely disease given a binary feature vector
  • Fetch description, diet, medications, precautions, and workouts
    for the predicted disease from CSV files in /data/
"""

import os
import numpy as np
import pandas as pd
import joblib
from functools import lru_cache


# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR  = os.path.dirname(os.path.dirname(__file__))   # project root
_MODEL_DIR = os.path.join(_BASE_DIR, "model")
_DATA_DIR  = os.path.join(_BASE_DIR, "data")

_MODEL_PATH        = os.path.join(_MODEL_DIR, "model.pkl")
_DESCRIPTIONS_PATH = os.path.join(_DATA_DIR,  "descriptions.csv")
_DIETS_PATH        = os.path.join(_DATA_DIR,  "diets.csv")
_MEDICATIONS_PATH  = os.path.join(_DATA_DIR,  "medications.csv")
_PRECAUTIONS_PATH  = os.path.join(_DATA_DIR,  "precautions.csv")
_WORKOUTS_PATH     = os.path.join(_DATA_DIR,  "workouts.csv")


# ── Model loading ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_model():
    """
    Load the trained RandomForest model from disk.
    Cached so the (potentially large) model file is only loaded once.
    """
    return joblib.load(_MODEL_PATH)


# ── CSV data loading (cached) ─────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_descriptions() -> pd.DataFrame:
    df = pd.read_csv(_DESCRIPTIONS_PATH)
    df["Disease"] = df["Disease"].str.lower().str.strip()
    return df

@lru_cache(maxsize=1)
def _load_diets() -> pd.DataFrame:
    df = pd.read_csv(_DIETS_PATH)
    df["Disease"] = df["Disease"].str.lower().str.strip()
    return df

@lru_cache(maxsize=1)
def _load_medications() -> pd.DataFrame:
    df = pd.read_csv(_MEDICATIONS_PATH)
    df["Disease"] = df["Disease"].str.lower().str.strip()
    return df

@lru_cache(maxsize=1)
def _load_precautions() -> pd.DataFrame:
    df = pd.read_csv(_PRECAUTIONS_PATH)
    df["Disease"] = df["Disease"].str.lower().str.strip()
    return df

@lru_cache(maxsize=1)
def _load_workouts() -> pd.DataFrame:
    df = pd.read_csv(_WORKOUTS_PATH)
    df["Disease"] = df["Disease"].str.lower().str.strip()
    return df


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_disease(feature_vector: np.ndarray) -> tuple[str, dict[str, float]]:
    """
    Predict the most likely disease from a binary symptom feature vector.

    Args:
        feature_vector: numpy array of shape (1, n_features)

    Returns:
        (predicted_disease_name, {disease: probability, ...})
        The probabilities dict contains the top-5 predictions.
    """
    from utils.preprocess import get_all_symptoms

    model = load_model()

    # Wrap in DataFrame so sklearn doesn't warn about missing feature names
    feat_cols = get_all_symptoms()
    df_input  = pd.DataFrame(feature_vector, columns=feat_cols)

    predicted = model.predict(df_input)[0]

    # Top-5 class probabilities
    proba = model.predict_proba(df_input)[0]
    classes = model.classes_
    top5_idx = np.argsort(proba)[::-1][:5]
    top5 = {classes[i]: round(float(proba[i]) * 100, 2) for i in top5_idx}

    return predicted, top5


# ── Information retrieval ─────────────────────────────────────────────────────

def _lookup_row(df: pd.DataFrame, disease: str) -> pd.Series | None:
    """Return the first matching row for the given disease name (case-insensitive)."""
    key = disease.lower().strip()
    rows = df[df["Disease"] == key]
    if rows.empty:
        return None
    return rows.iloc[0]


def get_description(disease: str) -> str:
    """Return the text description for the given disease."""
    row = _lookup_row(_load_descriptions(), disease)
    if row is None:
        return f"No detailed description available for '{disease}'. Please consult a healthcare professional."
    return str(row.get("Description", ""))


def get_diet(disease: str) -> list[str]:
    """Return a list of dietary recommendations for the given disease."""
    row = _lookup_row(_load_diets(), disease)
    if row is None:
        return ["Consult a nutritionist for personalised dietary advice."]
    cols = [f"Diet_{i}" for i in range(1, 6)]
    return [str(row[c]) for c in cols if c in row.index and str(row[c]).strip() not in ("", "nan")]


def get_medications(disease: str) -> list[str]:
    """Return a list of commonly prescribed medications for the given disease."""
    row = _lookup_row(_load_medications(), disease)
    if row is None:
        return ["Please consult a doctor for appropriate medication guidance."]
    cols = [f"Medication_{i}" for i in range(1, 6)]
    return [str(row[c]) for c in cols if c in row.index and str(row[c]).strip() not in ("", "nan")]


def get_precautions(disease: str) -> list[str]:
    """Return a list of precautions for the given disease."""
    row = _lookup_row(_load_precautions(), disease)
    if row is None:
        return ["Follow your doctor's advice and monitor your symptoms carefully."]
    cols = [f"Precaution_{i}" for i in range(1, 6)]
    return [str(row[c]) for c in cols if c in row.index and str(row[c]).strip() not in ("", "nan")]


def get_workouts(disease: str) -> list[str]:
    """Return a list of recommended exercises/workouts for the given disease."""
    row = _lookup_row(_load_workouts(), disease)
    if row is None:
        return ["Consult your doctor before starting any exercise programme."]
    cols = [f"Workout_{i}" for i in range(1, 6)]
    return [str(row[c]) for c in cols if c in row.index and str(row[c]).strip() not in ("", "nan")]


def get_full_disease_info(disease: str) -> dict:
    """
    Convenience function: return all supporting information for a disease
    as a single dictionary.

    Returns:
        {
            "description":  str,
            "diet":         list[str],
            "medications":  list[str],
            "precautions":  list[str],
            "workouts":     list[str],
        }
    """
    return {
        "description": get_description(disease),
        "diet":        get_diet(disease),
        "medications": get_medications(disease),
        "precautions": get_precautions(disease),
        "workouts":    get_workouts(disease),
    }
