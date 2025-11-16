from pathlib import Path


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