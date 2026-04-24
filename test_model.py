import __main__

from flamapy.metamodels.fm_metamodel.transformations import UVLReader
from flamapy.metamodels.z3_metamodel.transformations import FmToZ3
from flamapy.metamodels.pysat_metamodel.transformations import FmToPysat
from flamapy.metamodels.pysat_metamodel.operations import (
    PySATConfigurations,
    PySATConfigurationsNumber
)
from flamapy.metamodels.z3_metamodel.operations import (
    Z3Satisfiable,
    Z3Configurations,
    Z3ConfigurationsNumber,
    Z3CoreFeatures,
    Z3DeadFeatures,
    Z3FalseOptionalFeatures,
    Z3AttributeOptimization,
    Z3SatisfiableConfiguration
)


MODEL_PATH = 'resources/models/mutex.uvl'


def main() -> None:
    fm_model = UVLReader(MODEL_PATH).transform()
    z3_model = FmToZ3(fm_model).transform()
    pysat_model = FmToPysat(fm_model).transform()

    result = Z3ConfigurationsNumber().execute(z3_model).get_result()
    print(f'Number of configurations (Z3): {result}')

    result = PySATConfigurationsNumber().execute(pysat_model).get_result()
    print(f'Number of configurations (PySAT): {result}')


if __name__ == "__main__":
    main()