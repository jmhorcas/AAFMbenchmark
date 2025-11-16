# utils/benchmark_utils.py

import csv
import os
from pathlib import Path
from typing import Any

from flamapy.metamodels.fm_metamodel.transformations import UVLReader
from flamapy.metamodels.z3_metamodel.transformations import FmToZ3


KEY_MODEL_NAME = 'Model Name'
KEY_OPERATION = 'Operation'
KEY_RESULT = 'Result'
KEY_REPETITIONS = 'Repetitions'
KEY_TIME_MEAN = 'Time_Mean (s)'
KEY_TIME_MEDIAN = 'Time_Median (s)'
KEY_TIME_STDDEV = 'Time_StdDev (s)'
KEY_TIMEOUT = 'Timeout (s)'


CSV_HEADERS: list[str] = [
    KEY_MODEL_NAME, 
    KEY_OPERATION, 
    KEY_RESULT, 
    KEY_REPETITIONS, 
    KEY_TIME_MEAN, 
    KEY_TIME_MEDIAN, 
    KEY_TIME_STDDEV,
    KEY_TIMEOUT
]


def write_results_incrementally(filename: str, 
                                headers: list[str], 
                                data: list[dict[str, Any]]) -> None:
    """
    Writes results incrementally to a CSV file.
    If the file does not exist, it writes the headers ('w' mode). 
    Otherwise, it appends the data ('a' mode) without headers.
    """
    
    file_path = Path(filename)
    file_exists = file_path.exists()
    
    with file_path.open('a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerows(data)


def load_fm_model(model_path: str):
    """Loads and returns the Feature Model (FM) from a UVL file."""
    return UVLReader(model_path).transform()


def get_transformed_model(fm_model: Any, transformation_type: str) -> Any:
    """Applies the necessary transformation to the FM."""
    if transformation_type == 'FM':
        return fm_model
    if transformation_type == 'Z3':
        return FmToZ3(fm_model).transform()
    raise ValueError(f"Unknown transformation type: {transformation_type}")


def get_processed_model_names(filename: str) -> set[str]:
    """
    Reads the existing CSV file and returns a set of model names already processed.
    Returns an empty set if the file does not exist or is empty/corrupt.
    """
    file_path = Path(filename)
    if not file_path.exists():
        return set()

    processed_models: set[str] = set()
    try:
        with file_path.open('r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Ensure the required header for model name exists
            if KEY_MODEL_NAME not in reader.fieldnames:
                print(f"⚠️ Warning: CSV file '{filename}' exists but lacks the required header '{KEY_MODEL_NAME}'. Rerunning all models.")
                return set()

            for row in reader:
                # We collect all model names found in the file
                if row.get(KEY_MODEL_NAME):
                    processed_models.add(row[KEY_MODEL_NAME])
    except Exception as e:
        print(f"⚠️ Warning: Could not read existing CSV '{filename}' for processed models. Rerunning all models. Error: {e}")
        return set()
    
    return processed_models