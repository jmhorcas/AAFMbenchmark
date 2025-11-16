import os
import statistics
from typing import Any, Callable, Optional
from multiprocessing import Process, Manager, Queue
import time
from concurrent.futures import TimeoutError

from utils.timer import Timer 
from utils.file_manager import get_filepaths
from utils.benchmark_utils import (
    load_fm_model, 
    get_transformed_model,
    write_results_incrementally,
    get_processed_model_names,
    CSV_HEADERS,
    KEY_MODEL_NAME, KEY_OPERATION, KEY_RESULT, KEY_REPETITIONS, KEY_TIME_MEAN, KEY_TIME_MEDIAN, KEY_TIME_STDDEV, KEY_TIMEOUT
)

from flamapy.core.operations import Operation
from flamapy.metamodels.z3_metamodel.operations import (
    Z3Satisfiable,
    Z3CoreFeatures,
    Z3DeadFeatures,
    Z3FalseOptionalFeatures
)
from flamapy.metamodels.z3_metamodel.operations.interfaces import OptimizationGoal
from utils.custom_operations import CustomZ3ConfigurationsNumber


# Type definition for operation configuration:
# (Operation Class, Required Transformation, Number of Repetitions, Additional Kwargs)
OperationConfig = tuple[Callable[..., Operation], str, int, dict[str, Any], Optional[float]]


def _subprocess_entry_point(OpClass: Callable[..., Operation], model_path: str, required_transform: str, op_kwargs: dict[str, Any], managed_ref: Any, result_queue: Queue) -> None:
    """
    Función de entrada del subproceso. RECONSTRUYE el modelo y ejecuta la operación.
    Envía el resultado final a través de 'result_queue'.
    """
    
    # 1. RECONSTRUIR EL MODELO Z3 (Fix para Pickling)
    try:
        fm_model = load_fm_model(model_path)
        target_model = get_transformed_model(fm_model, required_transform)
    except Exception as e:
        # Enviar el error a la cola si falla la reconstrucción
        result_queue.put(Exception(f"Model reconstruction failed in subprocess: {e}"))
        return

    # 2. Configurar y ejecutar la operación
    try:
        op_instance = OpClass()
        
        # Inyectar el contador observable
        if OpClass == CustomZ3ConfigurationsNumber:
             op_instance.set_partial_count_ref(managed_ref)
        
        # Configurar la operación
        for setter_method, value in op_kwargs.items():
            setter = getattr(op_instance, f'set_{setter_method}')
            setter(value)
        
        final_result = op_instance.execute(target_model).get_result()
        
        # 3. Enviar el resultado exitoso a la cola
        result_queue.put(final_result)
        
    except Exception as e:
        # Enviar cualquier error de ejecución a la cola
        result_queue.put(e)


class BenchmarkRunner:
    """
    Encapsulates the logic for running the feature model analysis benchmark, 
    handling repetitions, statistics, result formatting, and incremental persistence.
    """
    
    # Master registry defining all available operations, transformations, repetitions, and kwargs
    MASTER_OPERATION_REGISTRY: dict[str, OperationConfig] = {
        # Standard Operations
        'Satisfiable': (Z3Satisfiable, 'Z3', 30, {}, None),
        'CoreFeatures': (Z3CoreFeatures, 'Z3', 30, {}, None),
        'DeadFeatures': (Z3DeadFeatures, 'Z3', 30, {}, None),
        'FalseOptionalFeatures': (Z3FalseOptionalFeatures, 'Z3', 30, {}, None),
        'ConfigurationsNumber': (CustomZ3ConfigurationsNumber, 'Z3', 30, {}, 60.0),  # 60s timeout
        
        # Optimization Operation (requires 'attributes' kwarg)
        # NOTE: Attributes must exist in the models being analyzed!
        # 'AttributeOptimization_MinKcalMaxPrice': (
        #     Z3AttributeOptimization, 
        #     'Z3', 
        #     30, 
        #     {'attributes': {
        #         'Price': OptimizationGoal.MAXIMIZE,
        #         'Kcal': OptimizationGoal.MINIMIZE
        #     }},
        #     60.0  # 60s timeout
        # ),
    }

    def __init__(self, models_dir: str, operation_filter: str, output_csv: str, extensions: list[str] = ['.uvl']):
        """
        Initializes the runner with configuration parameters.
        """
        self.models_dir = models_dir
        self.extensions = extensions
        self.output_csv = output_csv
        self.operations_to_run = self._get_operations_to_run(operation_filter)
        self.manager = Manager()

    def _get_operations_to_run(self, operation_filter: str) -> dict[str, OperationConfig]:
        """Filters the MASTER_OPERATION_REGISTRY based on the command-line argument."""
        if operation_filter.lower() == 'all':
            return self.MASTER_OPERATION_REGISTRY
        
        if operation_filter in self.MASTER_OPERATION_REGISTRY:
            return {operation_filter: self.MASTER_OPERATION_REGISTRY[operation_filter]}
        
        raise ValueError(f"Operation '{operation_filter}' not found in registry.")

    def _format_result(self, op_name: str, final_result: Any) -> str:
        """Helper method to format the operation result for the CSV."""
        if op_name.startswith('AttributeOptimization'):
            if final_result:
                num_configs = len(final_result)
                # The result is typically [(config, values_dict), ...]
                first_config_values = final_result[0][1] 
                value_summary = ', '.join(f'{k}={v}' for k, v in first_config_values.items())
                return f'{num_configs} configs, {value_summary}'
            return 'No optimum configurations'
        
        if isinstance(final_result, (list, set, tuple)):
            return f'{len(final_result)} elements'
            
        return str(final_result)

    def _direct_execution_path(self, OpClass: Callable[..., Operation], target_model: Any, op_kwargs: dict[str, Any], partial_count_ref: Optional[list[int]] = None) -> Any:
        """Ejecuta la operación directamente, usando el target_model ya cargado."""
        operation_instance = OpClass()
        
        # Inyectar el contador si es la clase observable
        if OpClass == CustomZ3ConfigurationsNumber and partial_count_ref is not None and hasattr(operation_instance, 'set_partial_count_ref'):
            operation_instance.set_partial_count_ref(partial_count_ref)
        
        # Configurar y ejecutar
        for setter_method, value in op_kwargs.items():
            setter = getattr(operation_instance, f'set_{setter_method}')
            setter(value)
        
        return operation_instance.execute(target_model).get_result()

    def _execute_op_with_config(self, 
                                OpClass: Callable[..., Operation], 
                                target_model: Any, 
                                op_kwargs: dict[str, Any], 
                                partial_count_ref: Optional[list[int]] = None) -> Any:
        """
        Ejecuta la operación directamente (sin ThreadPoolExecutor). 
        Utilizado para las ejecuciones sin timeout.
        """
        operation_instance = OpClass()

        # Inyectar el contador si es la clase observable
        if OpClass == CustomZ3ConfigurationsNumber and partial_count_ref is not None:
            # Usamos el hasattr() como un seguro, aunque ya sabemos que la clase lo tiene
            if hasattr(operation_instance, 'set_partial_count_ref'):
                operation_instance.set_partial_count_ref(partial_count_ref)
        
        # Configuración de atributos (e.g., set_attributes, set_partial_configuration)
        for setter_method, value in op_kwargs.items():
            setter = getattr(operation_instance, f'set_{setter_method}')
            setter(value)
        
        return operation_instance.execute(target_model).get_result()

    def _run_single_operation(self, 
                              model_path: str, 
                              fm_model: Any, 
                              op_name: str, 
                              OpClass: Callable[..., Operation], 
                              required_transform: str, 
                              n_reps: int, 
                              op_kwargs: dict[str, Any], 
                              op_timeout: Optional[float], 
                              transformed_models: dict[str, Any]) -> list[dict[str, Any]]:
        model_name = os.path.basename(model_path)
        times: list[float] = []
        final_result = None
        
        # 1. Get the required transformed model (solo para la ejecución directa)
        try:
            if required_transform not in transformed_models:
                transformed_models[required_transform] = get_transformed_model(fm_model, required_transform)
            target_model = transformed_models[required_transform] 
        except Exception as e:
            return self._build_error_row(model_name, op_name, n_reps, f"TRANSFORMATION ERROR: {e}", op_timeout)

        # 2. Repetition Loop
        for _ in range(n_reps):
            
            # Inicializar el contador: ManagedList si hay timeout y es la custom op, sino lista normal
            if op_timeout is not None and OpClass == CustomZ3ConfigurationsNumber:
                shared_ref = self.manager.list([0]) 
            else:
                shared_ref = [0] 

            try:
                # --- Path con Timeout (Gestión Manual de Process) ---
                if op_timeout is not None:
                    result_queue = Queue()
                    
                    # 1. Crear el proceso
                    process = Process(target=_subprocess_entry_point, 
                                        args=(OpClass, model_path, required_transform, op_kwargs, shared_ref, result_queue))
                    
                    with Timer(logger=None, enabled=True) as t:
                        process.start()
                        
                        # 2. Esperar el resultado o el timeout
                        process.join(timeout=op_timeout) 
                    
                    # 3. Analizar el resultado de la espera
                    if process.is_alive():
                        # Si el proceso sigue vivo, ha habido timeout
                        
                        # 4. 🚨 TERMINACIÓN FORZADA Y CONTEO PARCIAL
                        process.terminate() # Detiene el proceso (SIGTERM/SIGKILL)
                        process.join() # Espera la terminación (necesario para liberar recursos)
                        
                        raise TimeoutError # Lanzamos la excepción para el manejo uniforme

                    # Si llega aquí, el proceso terminó antes del timeout. Recuperar resultado.
                    current_result = result_queue.get(timeout=0.1) # Recuperar el resultado sin esperar
                    
                    # Si el resultado es una excepción lanzada en el subproceso, relanzarla
                    if isinstance(current_result, Exception):
                        raise current_result

                # --- Path Directo (No Timeout) ---
                else:
                    current_result = self._direct_execution_path(
                        OpClass, target_model, op_kwargs, 
                        shared_ref if OpClass == CustomZ3ConfigurationsNumber else None
                    )
                
                # Success
                times.append(t.elapsed_time)
                final_result = current_result

            except TimeoutError:
                print(f" -> 🛑 TIMEOUT detected in repetition for {OpClass}.")
                if OpClass == CustomZ3ConfigurationsNumber:
                    final_count = shared_ref[0] 
                    # 🎯 EL MENSAJE CON EL CONTEO PARCIAL PARA EL CSV
                    error_message_for_csv = f"{final_count}" 
                    print(f" -> 🛑 TIMEOUT: {op_name} exceeded {op_timeout}s. Partial count: {final_count}")
                else:
                    error_message_for_csv = "TIMEOUT"
                    print(f" -> 🛑 TIMEOUT: {op_name} exceeded {op_timeout}s.")
                # Devolver la fila de error inmediatamente, pasando el mensaje personalizado.
                # Esto corrige el bug.
                return self._build_error_row(model_name, op_name, n_reps, error_message_for_csv, op_timeout)

            except Exception as e:
                # Error en la ejecución (incluye errores de reconstrucción o errores Z3 internos)
                print(f" -> ⚠️ REPETITION ERROR in {op_name}: {e}")
                return self._build_error_row(model_name, op_name, n_reps, f"EXECUTION ERROR: {e}", op_timeout)

        # 3. Statistical Calculation (Lógica sin cambios)
        if not times:
             return self._build_error_row(model_name, op_name, n_reps, "EXECUTION ERROR", op_timeout)

        mean_time = statistics.mean(times)
        median_time = statistics.median(times)
        stddev_time = statistics.stdev(times) if len(times) > 1 else 0.0
        
        # 4. Prepare the row for the CSV (Lógica sin cambios)
        timeout_value_for_csv = 'None'
        row_data = {
            KEY_MODEL_NAME: model_name,
            KEY_OPERATION: op_name,
            KEY_RESULT: self._format_result(op_name, final_result),
            KEY_REPETITIONS: len(times),
            KEY_TIME_MEAN: mean_time,
            KEY_TIME_MEDIAN: median_time,
            KEY_TIME_STDDEV: stddev_time,
            KEY_TIMEOUT: timeout_value_for_csv,
        }

        print(f" -> {op_name} (N={len(times)}): Mean={mean_time:.4f}s, Median={median_time:.4f}s")
        return [row_data]


    def _build_error_row(self, 
                         model_name: str, 
                         op_name: str, 
                         n_reps: int, 
                         error_message: str, 
                         op_timeout: Optional[float]) -> list[dict[str, Any]]:
        """Helper to consistently build an error/timeout row."""
        
        # Si el mensaje de error comienza con "TIMEOUT" (incluyendo "TIMEOUT (Count: X)"),
        # mostramos el valor configurado del timeout. En cualquier otro caso, es None.
        if error_message.startswith("TIMEOUT"):
            timeout_value_for_csv = op_timeout 
        else:
            # Por ejemplo, si el error es de tipo "EXECUTION ERROR" o "TRANSFORMATION ERROR"
            timeout_value_for_csv = None 
            
        return [{
            KEY_MODEL_NAME: model_name,
            KEY_OPERATION: op_name,
            KEY_RESULT: error_message, # Usa el mensaje de error (incluyendo el conteo parcial)
            KEY_REPETITIONS: n_reps,
            KEY_TIME_MEAN: 0.0,
            KEY_TIME_MEDIAN: 0.0,
            KEY_TIME_STDDEV: 0.0,
            KEY_TIMEOUT: timeout_value_for_csv, # Valor corregido (float o None)
        }]


    def run_model_benchmark(self, model_path: str) -> list[dict[str, Any]]:
        """Executes all defined operations on a single model and collects results."""
        model_name = os.path.basename(model_path)
        all_results: list[dict[str, Any]] = []

        print(f"  ⚙️  Processing Model: {model_name}")

        try:
            fm_model = load_fm_model(model_path)
            transformed_models: dict[str, Any] = {KEY_MODEL_NAME: fm_model} # KEY_MODEL_NAME is 'Model Name', but should be 'FM' for clarity. Reverting to 'FM'.
            transformed_models: dict[str, Any] = {'FM': fm_model} 

        except Exception as e:
            print(f"⚠️ ERROR during initial Load/Transformation for {model_name}: {e}")
            return self._build_error_row(model_name, "Load/Transform", 1, f"ERROR: {e}")
        
        # Iterate over all defined operations
        for op_name, op_config in self.operations_to_run.items():
            OpClass, required_transform, n_reps, op_kwargs, op_timeout = op_config
            
            op_results = self._run_single_operation(
                model_path, fm_model, op_name, OpClass, required_transform, 
                n_reps, op_kwargs, op_timeout, transformed_models
            )
            all_results.extend(op_results)
            
        return all_results

    def run_full_benchmark(self) -> None:
        """
        Orchestrates the full benchmark across all models and operations, 
        writing results incrementally after each model is processed to ensure fault tolerance.
        """
        all_model_paths = get_filepaths(directory=self.models_dir, extensions_filter=self.extensions)
        
        if not all_model_paths:
            print(f"🛑 No models found in directory: {self.models_dir}")
            return

        processed_model_names = get_processed_model_names(self.output_csv)
        model_paths_to_run = []
        for path in all_model_paths:
            model_name = os.path.basename(path)
            if model_name not in processed_model_names:
                model_paths_to_run.append(path)

        num_total = len(all_model_paths)
        num_to_run = len(model_paths_to_run)
        num_skipped = num_total - num_to_run

        print(f"🔎 Found {num_total} models to analyze in total.")
        if num_skipped > 0:
            print(f"⏩ Skipping {num_skipped} models already processed and found in '{self.output_csv}'.")
        
        if not model_paths_to_run:
            print(f"✅ All models already processed. Nothing new to run.")
            return
        
        # Process all models
        for i, model_path in enumerate(model_paths_to_run):
            print(f"\n[PROGRESS: {i+1}/{num_to_run}] Analyzing {os.path.basename(model_path)}")
            
            # Run all operations for the current model
            model_results = self.run_model_benchmark(model_path)
            
            # --- FAULT TOLERANCE STEP: WRITE INCREMENTALLY ---
            # Results are saved immediately after completing a model's execution.
            if model_results:
                write_results_incrementally(self.output_csv, CSV_HEADERS, model_results)
                
        print("\n✅ Full Benchmark Process Completed.")