import argparse

from benchmark_runner import BenchmarkRunner


OUTPUT_CSV = 'benchmark_results_full.csv'
DEFAULT_EXTENSIONS = ['.uvl'] 


def parse_arguments():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Feature Model Analysis Benchmark Engine.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'models_dir',
        type=str,
        help='Directory containing the feature model files (e.g., resources/models/uvl_models).'
    )
    parser.add_argument(
        '--operation',
        type=str,
        default='all',
        help=f'Specific operation to benchmark. Use "all" to run all registered operations. '
             f'(See BenchmarkRunner.MASTER_OPERATION_REGISTRY for available names)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=OUTPUT_CSV,
        help='Name of the output CSV file.'
    )
    return parser.parse_args()


def main_benchmark() -> None:
    """Main function that orchestrates the entire benchmark process."""
    args = parse_arguments()
    
    models_dir = args.models_dir
    target_op_name = args.operation
    output_csv = args.output_file
    
    print(f"🚀 Starting Feature Model Benchmark")
    
    try:
        # 1. Initialize the Runner (Dependency Injection of configuration)
        # The runner handles the validation of the operation name and file persistence internally.
        runner = BenchmarkRunner(
            models_dir=models_dir, 
            operation_filter=target_op_name, 
            output_csv=output_csv, 
            extensions=DEFAULT_EXTENSIONS
        )

        # 2. Run the full benchmark (persistence is handled incrementally inside the runner)
        runner.run_full_benchmark() 

    except ValueError as e:
        # Catches the error if the user specifies an invalid operation name
        print(f"🛑 Configuration Error: {e}")
    except Exception as e:
        # Catch unexpected errors during setup or execution
        print(f"❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    main_benchmark()