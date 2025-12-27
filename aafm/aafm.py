import sys
import logging
from alive_progress import alive_bar
from utils.timer import Timer
from typing import Any
import statistics

from aafm import catalog
from flamapy.core.models import VariabilityModel
from flamapy.metamodels.fm_metamodel.transformations import UVLReader
from flamapy.metamodels.pysat_metamodel.transformations import FmToPysat
from flamapy.metamodels.bdd_metamodel.transformations import FmToBDD
from flamapy.metamodels.z3_metamodel.transformations import FmToZ3

from flamapy.metamodels.fm_metamodel.operations.fm_language_level import FMLanguageLevel, LanguageLevel


LOGGER = logging.getLogger(__name__)


class AAFM():

    def __init__(self, 
                 fm_path: str, 
                 progress_bar: bool = True,
                 precision: int = 4) -> None:
        self.fm_path = fm_path
        self.progress_bar = progress_bar
        self.precision = precision
        self.models: dict[catalog.Model, VariabilityModel] = {}
        self.read_fm(fm_path)

    def read_fm(self, fm_path: str) -> None:
        """Reads the feature model from the specified path."""
        LOGGER.debug(f'Reading FM from {fm_path} ...')
        try:
            with alive_bar(title=f'Reading FM from {fm_path} ...', disable=not self.progress_bar) as bar:
                with Timer(logger=None) as t:
                    self.models[catalog.Model.FM] = UVLReader(fm_path).transform()
                bar()
            LOGGER.debug(f'FM successfully read. Elapsed time (UVLReader): {t.elapsed_time:.{self.precision}f} s.')
        except Exception as e:
            LOGGER.error(f'Error reading FM from {fm_path}: {e}')
            sys.exit(1)

    def get_language_level(self) -> str:
        language_level = FMLanguageLevel().execute(self.models[catalog.Model.FM]).get_result()
        return f'{language_level.major.name.capitalize()} ({", ".join(minor.name.replace("_", " ").capitalize() for minor in language_level.minors)})'

    def transform(self, formalization: catalog.Formalization) -> None:
        """Transforms the feature model to the specified formalization."""
        LOGGER.debug(f'Transforming FM to {formalization.name} ...')
        try:
            with alive_bar(title=f'Transforming FM to {formalization.name} ...', disable=not self.progress_bar) as bar:
                with Timer(logger=None) as t:
                    self.models[catalog.MODELS[formalization]] = catalog.TRANSFORMATIONS[formalization](self.models[catalog.Model.FM]).transform()
                bar()
            LOGGER.debug(f'FM successfully transformed to {formalization.name}. Elapsed time ({catalog.TRANSFORMATIONS[formalization].__name__}): {t.elapsed_time:.{self.precision}f} s.')
        except Exception as e:
            LOGGER.warning(f'Error transforming FM to {formalization.name}: {e}')
            self.models[catalog.MODELS[formalization]] = None

    def execute_operation(self, 
                          operation: catalog.Operation, 
                          formalization: catalog.Formalization,
                          runs: int = 1,
                          ) -> str:
        """Executes the specified operation on the given formalization."""
        stats: dict[catalog.Stat, float] = {}
        elapsed_times: list[float] = []
        LOGGER.debug(f'Execute operation: {operation} on formalization: {formalization.name}. Runs: {runs}')
        with alive_bar(runs, title=f'Executing {runs} runs of {operation.name} on {formalization.name} ...', disable=not self.progress_bar) as bar:
            for run in range(1, runs + 1):
                with Timer(logger=None) as t:
                    result = catalog.FORMALIZATION_OPERATIONS[formalization][operation]().execute(self.models[catalog.MODELS[formalization]]).get_result()
                bar()
                elapsed_times.append(t.elapsed_time)
        stats[catalog.Stat.RUNS] = runs
        stats[catalog.Stat.MEAN] = statistics.mean(elapsed_times)
        stats[catalog.Stat.STDEV] = statistics.stdev(elapsed_times) if runs > 1 else 0.0
        stats[catalog.Stat.MEDIAN] = statistics.median(elapsed_times)
        LOGGER.debug(f'Operation {operation} on formalization {formalization.name} finished. Stats: '
                     f'Runs: {stats[catalog.Stat.RUNS]}, '
                     f'Mean: {stats[catalog.Stat.MEAN]:.{self.precision}f} s, '
                     f'Stdev: {stats[catalog.Stat.STDEV]:.{self.precision}f} s, '
                     f'Median: {stats[catalog.Stat.MEDIAN]:.{self.precision}f} s.')
        return result
        