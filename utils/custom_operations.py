# utils/custom_operations.py

from typing import Optional, List, cast
from flamapy.core.operations import Operation
from flamapy.core.models import VariabilityModel

import z3 
import logging
from flamapy.metamodels.fm_metamodel.models import FeatureType 
from flamapy.metamodels.z3_metamodel.models import Z3Model 
from flamapy.metamodels.configuration_metamodel.models import Configuration 

LOGGER = logging.getLogger(__name__)


# --- Lógica de Conteo Z3 Modificada (Observable) ---

def z3_solve_and_count_with_observation(model: Z3Model, 
                                        partial_configuration: Optional[Configuration], 
                                        partial_count_ref: List[int]) -> int:
    """
    Ejecuta el bucle de enumeración de Z3, actualizando la lista mutable 'partial_count_ref'
    con el número de configuraciones encontradas hasta el momento.
    """
    solver = z3.Solver(ctx=model.ctx)

    # 1. Add the model constraints to the solver
    solver.add(model.constraints)

    # 2. Create constraints for the given partial configuration (if any)
    if partial_configuration is not None:
        if partial_configuration.is_full:
            LOGGER.warning("Full configuration provided.")
            return 0  # No configurations
        config_constraints = []
        for feature_name, feature_value in partial_configuration.elements.items():
            if feature_name not in model.features:
                LOGGER.error(f"ERROR: the feature '{feature_name}' of the partial "\
                               "configuration does not exist in the Z3 model.")
                return 0
            feature_info = model.features[feature_name]
            constraints = model.create_feature_constraints(feature_value, # Asumo que es model.create_feature_constraints
                                                             feature_info, 
                                                             model.ctx)
            config_constraints.extend(constraints)
        solver.add(config_constraints)

    # 3. Enumerate all solutions
    configs_list = [] # Usamos 'configs_list' para no confundir con el return
    while solver.check() == z3.sat:
        m = solver.model()
        config_elements = {}
        block = []

        for feature, feature_info in model.features.items():
            sel = feature_info.sel
            selected = m.evaluate(sel, model_completion=True)
            block.append(sel != selected)  # block this value in the next iteration
            if feature_info.ftype == FeatureType.BOOLEAN:  # boolean feature
                value = z3.is_true(selected)
            else:  # typed feature
                if z3.is_true(selected):
                    val_expr = feature_info.val
                    if val_expr is None:
                        raise ValueError(f'Feature {feature} has no value expression.')
                    value = m.evaluate(val_expr, model_completion=True)
                    block.append(val_expr != value)  # block the value in the next iter.
                    if feature_info.ftype == FeatureType.INTEGER:
                        value = value.as_long()
                    elif feature_info.ftype == FeatureType.REAL:
                        value = value.as_decimal(model.DEFAULT_PRECISION) # Asumo que DEFAULT_PRECISION es de la clase Z3Model/Model
                    elif feature_info.ftype == FeatureType.STRING:
                        value = value.as_string()
                else:
                    value = False  # not selected
            config_elements[feature] = value
        
        config = Configuration(config_elements)
        configs_list.append(config)
        
        # 🎯 LÍNEA CRÍTICA: Actualizar el contador externo
        partial_count_ref[0] = len(configs_list)
        
        solver.add(z3.Or(block))  # block this solution
        
    return len(configs_list) # Devolver el conteo total


# --- Clase de Operación Personalizada ---

class CustomZ3ConfigurationsNumber(Operation):
    """
    Operación personalizada para contar configuraciones.
    Llama a la lógica observable y devuelve el resultado entero.
    """
    def __init__(self) -> None:
        self._result: int = 0
        self._partial_configuration: Optional[Configuration] = None
        self.partial_count_ref: List[int] = [0] # Referencia interna (será reemplazada por el runner)
        
    def set_partial_configuration(self, partial_configuration: Optional[Configuration]) -> None:
        self._partial_configuration = partial_configuration
        
    def set_partial_count_ref(self, partial_count_ref: List[int]) -> None:
        """Permite que el runner inyecte su propia referencia al contador mutable."""
        self.partial_count_ref = partial_count_ref

    def execute(self, model: VariabilityModel) -> 'CustomZ3ConfigurationsNumber':
        z3_model = cast(Z3Model, model)
        # Llama a la lógica Z3 observable y guarda el conteo final
        self._result = z3_solve_and_count_with_observation(z3_model, self._partial_configuration, self.partial_count_ref)
        return self

    def get_result(self) -> int:
        return self._result

    def get_configurations_number(self) -> int:
        return self.get_result()