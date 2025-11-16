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
