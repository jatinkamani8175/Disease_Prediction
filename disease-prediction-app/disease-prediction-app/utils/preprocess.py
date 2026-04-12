"""
utils/preprocess.py — Data preprocessing utilities.

Responsibilities:
  • Load and cache the list of all valid symptom names from feature_columns.csv
  • Convert a list of user-selected symptoms into a binary feature vector
    that matches the exact column order the ML model was trained on.
"""

import os
import pandas as pd
import numpy as np
from functools import lru_cache


# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR  = os.path.dirname(os.path.dirname(__file__))   # project root
_MODEL_DIR = os.path.join(_BASE_DIR, "model")


@lru_cache(maxsize=1)
def get_all_symptoms() -> list[str]:
    """
    Load the ordered list of symptom column names from feature_columns.csv.
    Cached so the file is only read once per process lifetime.

    Returns:
        A list of raw symptom names exactly as the model was trained on,
        e.g. ['anxiety_and_nervousness', 'depression', ...].
    """
    path = os.path.join(_MODEL_DIR, "feature_columns.csv")
    series = pd.read_csv(path, header=None).squeeze()
    return series.tolist()


def get_display_symptoms() -> list[str]:
    """
    Return symptom names formatted for human-readable display
    (underscores → spaces, title-cased).

    Keeps the same ordering as get_all_symptoms() so index lookups work.
    """
    return [s.replace("_", " ").title() for s in get_all_symptoms()]


def symptoms_to_feature_vector(selected_display_symptoms: list[str]) -> np.ndarray:
    """
    Convert a list of display-friendly symptom strings into a binary
    feature vector that the trained ML model expects.

    Args:
        selected_display_symptoms: symptom names as shown in the UI,
            e.g. ['Anxiety And Nervousness', 'Fever'].

    Returns:
        A (1, n_features) numpy array of dtype int8 with 1s for selected
        symptoms and 0s elsewhere.
    """
    all_raw      = get_all_symptoms()          # canonical model feature order
    all_display  = get_display_symptoms()      # parallel display names

    # Build a quick lookup: display name → index in feature vector
    display_to_idx = {display: idx for idx, display in enumerate(all_display)}

    vector = np.zeros(len(all_raw), dtype=np.int8)

    for sym in selected_display_symptoms:
        idx = display_to_idx.get(sym)
        if idx is not None:
            vector[idx] = 1

    return vector.reshape(1, -1)


def validate_symptom_count(selected: list[str], minimum: int = 2) -> tuple[bool, str]:
    """
    Validate that the user selected at least `minimum` symptoms.

    Returns:
        (True, "")  if valid.
        (False, error_message)  otherwise.
    """
    if len(selected) < minimum:
        return False, f"Please select at least {minimum} symptoms for a reliable prediction."
    return True, ""
