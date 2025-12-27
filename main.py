import log_config
from dataclasses import dataclass
import logging
import argparse
from aafm.aafm import AAFM
from aafm import catalog


@dataclass
class ConfigParams:
    """Class for keeping track of configuration parameters."""
    progress_bar: bool
    log_file: str
    verbose: bool
    precision: int
    runs: int
    formalization: catalog.Formalization | None
    operation: catalog.Operation | None
    fm_path: str



def main(config_params: ConfigParams) -> None:
    LOGGER.info("Aplicación iniciada")
    aafm = AAFM(config_params.fm_path, 
                progress_bar=config_params.progress_bar, 
                precision=config_params.precision)
    aafm.transform(config_params.formalization or catalog.Formalization.Z3)
    print(f'Language Level: {aafm.get_language_level()}')

    result = aafm.execute_operation(operation=config_params.operation or catalog.Operation.SATISFIABLE,
                           formalization=config_params.formalization or catalog.Formalization.Z3,
                           runs=config_params.runs)
    print(f'Result of SATISFIABLE on Z3: {result}')

    LOGGER.info("Aplicación finalizada")


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AAFM', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('fm_model_path', type=str, help='Feature model file.')
    parser.add_argument('operation', type=str, default=None, help=f'Specific operation to benchmark. Options: {[op.name for op in catalog.Operation]}. Default: all operations.')
    parser.add_argument('-f', '--formalization', type=str, default=None, help=f'Specific formalization to benchmark. Options: {[f.name for f in catalog.Formalization]}. Default: all formalizations.')
    parser.add_argument('-r', '--runs', type=int, default=1, help='Number of runs to execute. Default: 1.')
    parser.add_argument('-pb', '--progress-bar', action='store_true', help='Show progress bar.')
    parser.add_argument('-p', '--precision', type=int, default=4, help='Number of decimal places for time reporting. Default: 4.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging.')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output file to save results. Default: None (print to console).')
    parser.add_argument('--log-file', dest='log_file', type=str, default=None, help='Log file path.')

    args = parser.parse_args()
    config_params = ConfigParams(
        progress_bar=args.progress_bar,
        log_file=args.log_file,
        verbose=args.verbose,
        precision=args.precision,
        runs=args.runs,
        formalization=catalog.Formalization[args.formalization.upper()] if args.formalization is not None else None,
        operation=catalog.Operation[args.operation.upper()] if args.operation is not None else None,
        fm_path=args.fm_model_path
    )
    if config_params.verbose:
        log_config.setup_logging(log_to_file=args.log_file is not None, filename=args.log_file)
    LOGGER = logging.getLogger(__name__)

    main(config_params)