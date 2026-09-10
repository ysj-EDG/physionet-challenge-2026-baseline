"""Minimal dataset I/O helpers required by the portable feature package."""

from typing import Dict, Tuple

import edfio
import numpy as np
import pandas as pd


DEMOGRAPHICS_FILE = "demographics.csv"
PHYSIOLOGICAL_DATA_SUBFOLDER = "physiological_data"
ALGORITHMIC_ANNOTATIONS_SUBFOLDER = "algorithmic_annotations"

HEADERS = {
    "site_id": "SiteID",
    "patient_id": "BDSPPatientID",
    "creation_time": "CreationTime",
    "bids_folder": "BidsFolder",
    "session_id": "SessionID",
    "age": "Age",
    "sex": "Sex",
    "race": "Race",
    "ethnicity": "Ethnicity",
    "bmi": "BMI",
    "time_to_event": "Time_to_Event",
    "label": "Cognitive_Impairment",
    "last_visit_date": "Last_Known_Visit_Date",
    "time_to_last_visit": "Time_to_Last_Visit",
}


def load_signal_data(edf_path, return_metadata=False):
    """Load EDF signals, optionally returning non-model alignment metadata.

    The default two-value return is intentionally unchanged for legacy callers.
    """
    edf = edfio.read_edf(str(edf_path), lazy_load_data=False)
    channel_dict = {}
    fs_dict = {}
    lengths = {}
    dimensions = {}
    for signal in edf.signals:
        label = signal.label.lower().strip()
        channel_dict[label] = signal.data
        fs_dict[label] = float(signal.sampling_frequency)
        lengths[label] = int(np.asarray(signal.data).size)
        dimensions[label] = str(signal.physical_dimension)
    if not return_metadata:
        return channel_dict, fs_dict

    def _safe_header(name):
        try:
            value = getattr(edf, name)
            return None if value is None else str(value)
        except Exception:
            return None

    return channel_dict, fs_dict, {
        "duration_sec": float(edf.duration),
        "startdate": _safe_header("startdate"),
        "starttime": _safe_header("starttime"),
        "signal_lengths": lengths,
        "physical_dimensions": dimensions,
    }


def _normalize_identifier(value) -> str:
    """Return a stable representation for identifiers loaded from JSON or CSV."""
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _identifier_matches(series: pd.Series, value) -> pd.Series:
    expected = _normalize_identifier(value)
    return series.map(_normalize_identifier) == expected


def load_demographics(metadata_file, patient_id, session_id):
    dataframe = pd.read_csv(
        metadata_file,
        dtype={
            HEADERS["bids_folder"]: "string",
            HEADERS["session_id"]: "string",
        },
    )
    mask = (
        _identifier_matches(dataframe[HEADERS["bids_folder"]], patient_id)
        & _identifier_matches(dataframe[HEADERS["session_id"]], session_id)
    )
    rows = dataframe.loc[mask]
    identity = (
        f"{HEADERS['bids_folder']}={_normalize_identifier(patient_id)!r}, "
        f"{HEADERS['session_id']}={_normalize_identifier(session_id)!r}"
    )
    if rows.empty:
        raise KeyError(f"demographics row not found: {identity}")
    if len(rows) > 1:
        raise ValueError(
            f"multiple demographics rows found ({len(rows)}): {identity}"
        )
    return rows.iloc[0].to_dict()


def load_diagnoses(metadata_file, patient_id):
    dataframe = pd.read_csv(
        metadata_file,
        dtype={HEADERS["bids_folder"]: "string"},
    )
    mask = _identifier_matches(dataframe[HEADERS["bids_folder"]], patient_id)
    values = dataframe.loc[mask, HEADERS["label"]].values
    if len(values) == 0:
        raise KeyError(f"patient not found in demographics: {patient_id}")
    return 1 if values[0] else 0


def load_age(data):
    value = data.get(HEADERS["age"])
    try:
        return float(value) if value is not None else 0.0
    except (ValueError, TypeError):
        return 0.0


def load_sex(data):
    value = str(data.get(HEADERS["sex"], "")).lower()
    if value.startswith("f"):
        return "Female"
    if value.startswith("m"):
        return "Male"
    return "Unknown"


def load_bmi(data):
    value = data.get(HEADERS["bmi"])
    try:
        number = float(value)
        return number if not np.isnan(number) else 0.0
    except (ValueError, TypeError):
        return 0.0


def get_standardized_race(data):
    value = str(data.get(HEADERS["race"], "")).lower()
    if any(word in value for word in ("white", "caucasian")):
        return "White"
    if any(word in value for word in ("black", "african american")):
        return "Black"
    if "asian" in value:
        return "Asian"
    unavailable = (
        "unknown", "unavailable", "declined", "unreported", "nan", "none",
        "not specified", "prefer not to say",
    )
    if not value.strip() or any(word == value or word in value for word in unavailable):
        return "Unavailable"
    return "Others"


__all__ = [
    "ALGORITHMIC_ANNOTATIONS_SUBFOLDER",
    "DEMOGRAPHICS_FILE",
    "HEADERS",
    "PHYSIOLOGICAL_DATA_SUBFOLDER",
    "get_standardized_race",
    "load_age",
    "load_bmi",
    "load_demographics",
    "load_diagnoses",
    "load_sex",
    "load_signal_data",
]
