from enum import Enum

from flamapy.metamodels.pysat_metamodel.transformations import FmToPysat
from flamapy.metamodels.pysat_metamodel.operations import (
    PySATSatisfiable,
    PySATCoreFeatures,
    PySATDeadFeatures,
    PySATFalseOptionalFeatures
)
from flamapy.metamodels.bdd_metamodel.transformations import FmToBDD
from flamapy.metamodels.bdd_metamodel.operations import (
    BDDSatisfiable,
    BDDCoreFeatures,
    BDDDeadFeatures,
    #BDDFalseOptionalFeatures
)
from flamapy.metamodels.z3_metamodel.transformations import FmToZ3
from flamapy.metamodels.z3_metamodel.operations import (
    Z3Satisfiable,
    Z3CoreFeatures,
    Z3DeadFeatures,
    Z3FalseOptionalFeatures
)


class Model(Enum):
    FM = 'fm_model'
    SAT = 'sat_model'
    BDD = 'bdd_model'
    Z3 = 'z3_model'


class Formalization(Enum):
    SAT = 'sat'
    BDD = 'bdd'
    Z3 = 'z3'


TRANSFORMATIONS = {
    Formalization.SAT: FmToPysat,
    Formalization.BDD: FmToBDD,
    Formalization.Z3: FmToZ3
}

MODELS = {
    Formalization.SAT: Model.SAT,
    Formalization.BDD: Model.BDD,
    Formalization.Z3: Model.Z3
}


class Operation(Enum):
    SATISFIABLE = 'satisfiable'
    CORE_FEATURES = 'core_features'
    DEAD_FEATURES = 'dead_features'
    FALSE_OPTIONAL_FEATURES = 'false_optional_features'


SAT_OPERATIONS = {
    Operation.SATISFIABLE: PySATSatisfiable,
    Operation.CORE_FEATURES: PySATCoreFeatures,
    Operation.DEAD_FEATURES: PySATDeadFeatures,
    Operation.FALSE_OPTIONAL_FEATURES: PySATFalseOptionalFeatures
}

BDD_OPERATIONS = {
    Operation.SATISFIABLE: BDDSatisfiable,
    Operation.CORE_FEATURES: BDDCoreFeatures,
    Operation.DEAD_FEATURES: BDDDeadFeatures,
    #Operation.FALSE_OPTIONAL_FEATURES: BDDFalseOptionalFeatures
}

Z3_OPERATIONS = {
    Operation.SATISFIABLE: Z3Satisfiable,
    Operation.CORE_FEATURES: Z3CoreFeatures,
    Operation.DEAD_FEATURES: Z3DeadFeatures,
    Operation.FALSE_OPTIONAL_FEATURES: Z3FalseOptionalFeatures
}

FORMALIZATION_OPERATIONS = {
    Formalization.SAT: SAT_OPERATIONS,
    Formalization.BDD: BDD_OPERATIONS,
    Formalization.Z3: Z3_OPERATIONS
}

DEFAULT_OPERATIONS = {
    Operation.SATISFIABLE: Z3Satisfiable,
    Operation.CORE_FEATURES: Z3CoreFeatures,
    Operation.DEAD_FEATURES: Z3DeadFeatures,
    Operation.FALSE_OPTIONAL_FEATURES: Z3FalseOptionalFeatures
}


class Stat(Enum):
    RUNS = 'runs'
    MEAN = 'mean'
    STDEV = 'stdev'
    MEDIAN = 'median'

