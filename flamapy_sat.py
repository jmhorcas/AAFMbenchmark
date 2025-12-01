import statistics
import argparse
import multiprocessing
import logging
import csv
import time
import sys
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import Any, Optional, Callable

from flamapy.metamodels.fm_metamodel.transformations import UVLReader, FlatFM
#from flamapy.metamodels.z3_metamodel.transformations import FmToZ3
#from flamapy.metamodels.z3_metamodel.operations import Z3Satisfiable
from flamapy.metamodels.pysat_metamodel.transformations import FmToPysat
from flamapy.metamodels.pysat_metamodel.operations import PySATSatisfiable


N_RUNS = 1
PRECISION = 4
OUTPUT_CSV = f'output.csv'
TIMEOUT = 60

# Constants for CSV headers
KEY_MODEL_NAME = 'Model Name'
KEY_OPERATION = 'Operation'
KEY_RESULT = 'Result'
KEY_RUNS = 'Runs'
KEY_TIME_MEAN = 'Time_Mean (s)'
KEY_TIME_MEDIAN = 'Time_Median (s)'
KEY_TIME_STDDEV = 'Time_StdDev (s)'
KEY_TIMEOUT = 'Timeout (s)'

CSV_HEADERS: list[str] = [
    KEY_MODEL_NAME, 
    KEY_OPERATION, 
    KEY_RESULT, 
    KEY_RUNS, 
    KEY_TIME_MEAN, 
    KEY_TIME_MEDIAN, 
    KEY_TIME_STDDEV,
    KEY_TIMEOUT
]


##################################################
# Configure logging
##################################################
def configure_logging(logfile=None, nolog=False, level=logging.DEBUG):
    if nolog:
        logging.disable(logging.CRITICAL)
        return

    if logfile:
        logging.basicConfig(
            filename=logfile,
            level=level,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
    else:
        logging.basicConfig(
            stream=sys.stdout,
            level=level,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
##################################################


##################################################
# Utils for timing code execution
##################################################
class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class."""


@dataclass
class Timer(ContextDecorator):
    """Time your code using a class, context manager, or decorator."""
    
    # Configuration parameters
    name: Optional[str] = None
    message: str = ""
    text: str = "Elapsed time: {:0.4f} s"  # Base format in seconds
    logger: Optional[Callable[[str], None]] = print
    enabled: bool = True
    
    # Internal attributes (instance only)
    _start_time: Optional[int] = field(default=None, init=False, repr=False)
    _elapsed_time_sec: float = field(default=0.0, init=False, repr=False)

    def start(self) -> None:
        """Start a new timer."""
        if not self.enabled:
            return

        if self._start_time is not None:
            raise TimerError("Timer is already running. Use .stop() to stop it.")

        # Usamos perf_counter_ns para una medición de tiempo de alta resolución para benchmarking
        self._start_time = time.perf_counter_ns()

    def stop(self) -> float:
        """Stop the timer and report the elapsed time."""
        if not self.enabled:
            return 0.0

        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it.")

        # Calculate the elapsed time
        end_time = time.perf_counter_ns()
        elapsed_ns = end_time - self._start_time
        
        # Store and return the time in seconds (standard unit)
        self._elapsed_time_sec = elapsed_ns * 1e-9
        self._start_time = None

        # --- Formatting and Logging Logic ---
        
        msg = f'{self.message} {self.text.format(self._elapsed_time_sec)}'
        
        # Optional format for minutes/hours (only for the log, not the return)
        if self._elapsed_time_sec >= 3600:
            elapsed_time_hour = self._elapsed_time_sec / 3600
            msg = f'{msg} ({elapsed_time_hour:.2f} h).'
        elif self._elapsed_time_sec >= 60:
            elapsed_time_min = self._elapsed_time_sec / 60
            msg = f'{msg} ({elapsed_time_min:.2f} m).'
        else:
            msg = f'{msg}.'

        # Report the elapsed time
        if self.logger:
            self.logger(msg)

        return self._elapsed_time_sec

    # Context Manager Methods
    
    def __enter__(self) -> "Timer":
        """Start a new timer as a context manager."""
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Stop the timer as a context manager."""
        if self.enabled:
            self.stop()
            
    # Helper method to get the result outside the context
    @property
    def elapsed_time(self) -> float:
        """Return the measured elapsed time."""
        return self._elapsed_time_sec
##################################################


##################################################
# Utils for file management
##################################################
def get_filepaths(directory: str, extensions_filter: list[str] = None) -> list[str]:
    """
    Obtains all filepaths of files with the given extensions from the specified 
    directory and its subdirectories.

    :param directory: The root directory as a string or Path object.
    :param extensions_filter: Optional list of file extensions (e.g., ['.uvl', '.json']).
    :return: List of filepaths as strings.
    """
    root_path = Path(directory)
    filepaths = []
    
    if extensions_filter is None:
        extensions_filter = []
    
    norm_filters = [ext.lower() for ext in extensions_filter]
    for path in root_path.rglob('*'):
        if path.is_file():
            if not norm_filters or any(path.name.lower().endswith(ext) for ext in norm_filters):
                filepaths.append(str(path))
    return filepaths


def get_models_to_run(models_dir: str, output_csv: str) -> list[str]:
    all_model_paths = get_filepaths(models_dir, ['.uvl'])
    if not all_model_paths:
        logging.error(f"🛑 No models found in directory: {models_dir}")
        return []
    processed_model_names = get_processed_model_names(output_csv)
    model_paths_to_run = []
    for path in all_model_paths:
        model_name = Path(path).stem
        if model_name not in processed_model_names:
            model_paths_to_run.append(path)
    num_total = len(all_model_paths)
    num_to_run = len(model_paths_to_run)
    num_skipped = num_total - num_to_run

    if num_skipped > 0:
        logging.info(f"⏩ Skipping {num_skipped} models already processed and found in '{output_csv}'.")
    if not model_paths_to_run:
        logging.info(f"✅ All models already processed. Nothing new to run.")
        return []
    return model_paths_to_run


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
                logging.warning(f"⚠️ Warning: CSV file '{filename}' exists but lacks the required header '{KEY_MODEL_NAME}'. Rerunning all models.")
                return set()

            for row in reader:
                # We collect all model names found in the file
                if row.get(KEY_MODEL_NAME):
                    processed_models.add(row[KEY_MODEL_NAME])
    except Exception as e:
        logging.warning(f"⚠️ Warning: Could not read existing CSV '{filename}' for processed models. Rerunning all models. Error: {e}")
        return set()
    
    return processed_models


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
##################################################


##################################################
# Utils for operation execution
##################################################
def worker_execute(op_class: Callable, fm_path: str, shared_data: DictProxy) -> None:
    try:
        fm_model = UVLReader(fm_path).transform()
        # Flat the FM
        flat_op = FlatFM(fm_model)
        flat_op.set_maintain_namespaces(False)
        fm_model = flat_op.transform()
    except Exception as e:
        shared_data['status'] = f'ERROR: reading FM model from {fm_path}: {e}'
        return

    try:
        #z3_model = FmToZ3(fm_model).transform()
        sat_model = FmToPysat(fm_model).transform()
    except Exception as e:
        shared_data['status'] = f'ERROR: transforming FM to SAT model from {fm_path}: {e}'
        return

    op = op_class()
    try:
        with Timer(logger=None) as timer:
            op.execute(sat_model)
        result = op.get_result()
        shared_data['final_result'] = result
        shared_data['elapsed_time'] = timer.elapsed_time
        shared_data['status'] = 'COMPLETED'
    except Exception as e:
        shared_data['status'] = f'ERROR: Execution failed for {op.__class__.__name__}: {e}'


def execute_operation_on_model(model_name: str,
                               fm_path: str,
                               Operation: Callable,
                               n_runs: int,
                               timeout: int,
                               precision: int) -> Optional[dict[str, Any]]:
    times: list[float] = []
    result = None
    is_timeout = False
    for _ in range(n_runs):
        try:
            with multiprocessing.Manager() as manager:
                shared_data = manager.dict({
                    'status': 'RUNNING',
                    'partial_result': 0,
                    'final_result': None
                })
                process = multiprocessing.Process(target=worker_execute, args=(Operation, 
                                                                               fm_path, 
                                                                               shared_data))
                process.start()
                process.join(timeout=timeout)  # Wait for completion or timeout
                if process.is_alive():  # Timeout occurred
                    process.terminate()  # Stop the process (SIGTERM/SIGKILL)
                    process.join()  # Wait for termination (necessary to free resources)

                    # Report timeout status and partial result
                    result = shared_data.get('partial_result', None)
                    shared_data['status'] = 'TIMEOUT'
                    is_timeout = True
                    break  # No need to continue repetitions on timeout
                if shared_data['status'] == 'COMPLETED':
                    result = shared_data['final_result']
                    result = len(result) if isinstance(result, list) else result
                    times.append(shared_data['elapsed_time'])
                elif shared_data['status'].startswith('ERROR'):
                    result = shared_data['status']
                    logging.error(f'❌ Worker Error executing {Operation.__name__} on model from {model_name}: {result}')
                    break  # Stop on error
        except Exception as e:
            logging.error(f'❌ Error executing {Operation.__name__} on model from {model_name}: {e}')
            return
    if times:
        mean_time = round(statistics.mean(times), precision)
        median_time = round(statistics.median(times), precision)
        stddev_time = round(statistics.stdev(times), precision) if len(times) > 1 else 0.0
    else:
        mean_time = median_time = stddev_time = 0.0
    row_data = {
            KEY_MODEL_NAME: model_name,
            KEY_OPERATION: Operation.__name__,
            KEY_RESULT: str(result),
            KEY_RUNS: len(times),
            KEY_TIME_MEAN: mean_time,
            KEY_TIME_MEDIAN: median_time,
            KEY_TIME_STDDEV: stddev_time,
            KEY_TIMEOUT: str(timeout) if is_timeout else str(None)
        }
    return row_data
##################################################


def analyze_model(fm_path: str,
                  operation: Callable,
                  n_runs: int,
                  timeout: int,
                  precision: int,
                  output_csv: str) -> None:
    model_name = Path(fm_path).stem
    result = execute_operation_on_model(model_name, fm_path, operation, n_runs, timeout, precision)
    if result:
        write_results_incrementally(output_csv, CSV_HEADERS, [result])   


def main():
    parser = argparse.ArgumentParser(description='Flamapy SAT: Execute the SAT operation over feature models.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('models_dir', type=str, help='Directory containing the feature model files in UVL.')
    parser.add_argument('-n', type=int, dest='runs', default=N_RUNS, help=f'Number of runs (default: {N_RUNS}).')
    parser.add_argument('-t', type=float, dest='timeout', default=TIMEOUT, help=f'Timeout in seconds (default: {TIMEOUT}).')
    parser.add_argument('-p', type=int, dest='precision', default=PRECISION, help=f'Precision of the timing results (default: {PRECISION}).')
    parser.add_argument('-o', type=str, dest='output_file', default=OUTPUT_CSV, help=f'Name of the output CSV file (default: {OUTPUT_CSV}).')
    parser.add_argument("--logfile", type=str, help="write logs to a file instead of console")
    parser.add_argument("--nolog", action="store_true", help="disable all logging")
    args = parser.parse_args()
    
    configure_logging(args.logfile, args.nolog)
    
    models_dir = args.models_dir
    runs = args.runs
    timeout = args.timeout
    precision = args.precision
    output_csv = args.output_file

    logging.info(f"🚀 Starting Flamapy SAT")

    model_paths_to_run = get_models_to_run(models_dir, output_csv)
    num_to_run = len(model_paths_to_run)
    logging.info(f"🔍 Found {num_to_run} models to analyze in '{models_dir}'.")
    for i, model_path in enumerate(model_paths_to_run):
        logging.info(f"[PROGRESS: {i+1}/{num_to_run}] Analyzing {Path(model_path).name}...")
        analyze_model(model_path, PySATSatisfiable, runs, timeout, precision, output_csv)
        
    logging.info("✅ Experiments completed.")


if __name__ == "__main__":
    main()