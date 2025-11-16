import statistics
import argparse
from pathlib import Path
from typing import Any, Optional, Callable

from flamapy.core.operations import Operation
from flamapy.metamodels.fm_metamodel.transformations import UVLReader
from flamapy.metamodels.z3_metamodel.models import Z3Model
from flamapy.metamodels.z3_metamodel.transformations import FmToZ3
from flamapy.metamodels.z3_metamodel.operations import (
    Z3Satisfiable,
    Z3CoreFeatures,
    Z3DeadFeatures,
    Z3FalseOptionalFeatures
)

from utils.timer import Timer
from utils.file_manager import get_models_to_run
from utils.benchmark_utils import (
    write_results_incrementally,
    CSV_HEADERS,
    KEY_MODEL_NAME, 
    KEY_OPERATION, 
    KEY_RESULT, 
    KEY_REPETITIONS, 
    KEY_TIME_MEAN, 
    KEY_TIME_MEDIAN, 
    KEY_TIME_STDDEV,
    KEY_TIMEOUT
)


N_REPETITIONS = 30
PRECISION = 4
OUTPUT_CSV = f'z3_benchmark_results.csv'
TIMEOUT = None
OPERATIONS = {'Satisfiable': Z3Satisfiable, 
              'CoreFeatures': Z3CoreFeatures, 
              'DeadFeatures': Z3DeadFeatures, 
              'FalseOptionalFeatures': Z3FalseOptionalFeatures}


def analyze_model(fm_path: str,
                  operations: list[Callable],
                  n_reps: int = N_REPETITIONS) -> None:
    model_name = Path(fm_path).stem
    try:
        fm_model = UVLReader(fm_path).transform()
    except Exception as e:
        print(f'❌ Error reading FM model from {fm_path}: {e}')

    try:
        z3_model = FmToZ3(fm_model).transform()
    except Exception as e:
        print(f'❌ Error transforming FM to Z3 model from {fm_path}: {e}')
        return
    for operation in operations:
        result = execute_operation_on_model(model_name, z3_model, operation, n_reps)
        if result:
            write_results_incrementally(OUTPUT_CSV, CSV_HEADERS, [result])   


def execute_operation_on_model(model_name: str,
                               z3_model: Z3Model,
                               operation: Callable,
                               n_reps: int = N_REPETITIONS) -> Optional[dict[str, Any]]:
    times: list[float] = []
    result = None
    for _ in range(n_reps):
        try:
            op = operation()
            with Timer(logger=None) as timer:
                op.execute(z3_model)
            result = op.get_result()
            times.append(timer.elapsed_time)
        except Exception as e:
            print(f'❌ Error executing {operation.__name__} on model from {model_name}: {e}')
            return
    mean_time = round(statistics.mean(times), PRECISION)
    median_time = round(statistics.median(times), PRECISION)
    stddev_time = round(statistics.stdev(times), PRECISION) if len(times) > 1 else 0.0
    
    row_data = {
            KEY_MODEL_NAME: model_name,
            KEY_OPERATION: operation.__name__,
            KEY_RESULT: 'True' if result else 'False',
            KEY_REPETITIONS: len(times),
            KEY_TIME_MEAN: mean_time,
            KEY_TIME_MEDIAN: median_time,
            KEY_TIME_STDDEV: stddev_time,
            KEY_TIMEOUT: str(TIMEOUT)
        }
    return row_data


def get_operations_to_run(operation_filter: str) -> list[Operation]:
    if operation_filter.lower() == 'all':
        return list(OPERATIONS.values())
    if operation_filter in OPERATIONS:
        return [OPERATIONS[operation_filter]]
    raise ValueError(f"🛑 Operation '{operation_filter}' not found in registry." \
                     f" Available operations: {list(OPERATIONS.keys())}")


def main():
    parser = argparse.ArgumentParser(description='AAFM Analysis', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('models_dir', type=str, help='Directory containing the feature model files (e.g., resources/models/uvl_models).')
    parser.add_argument('--operation', type=str, default='all', help=f'Specific operation to benchmark. Use "all" to run all registered operations. 'f'(See BenchmarkRunner.MASTER_OPERATION_REGISTRY for available names)')
    parser.add_argument('--output_file', type=str, default=OUTPUT_CSV, help='Name of the output CSV file.')
    
    args = parser.parse_args()
    models_dir = args.models_dir
    target_op_name = args.operation
    output_csv = args.output_file

    print(f"🚀 Starting Feature Model Benchmark")

    operations_to_run = get_operations_to_run(target_op_name)
    model_paths_to_run = get_models_to_run(models_dir, output_csv)
    num_to_run = len(model_paths_to_run)
    for i, model_path in enumerate(model_paths_to_run):
        print(f"[PROGRESS: {i+1}/{num_to_run}] Analyzing {Path(model_path).name}...")
        analyze_model(model_path, operations_to_run)
        
        
    print("✅ Full Benchmark Process Completed.")


if __name__ == "__main__":
    main()