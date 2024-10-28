import os
from pathlib import Path


def get_library_path():
    home = Path.home()
    library_dir_name = "transopt_files"
    library_path = home / library_dir_name

    if not library_path.exists():
        library_path.mkdir(parents=True, exist_ok=True)

    return library_path

def get_absolut_path():
    lib_path = get_library_path()
    absolut_dir_name = "Absolut"
    absolut_path = lib_path / absolut_dir_name
    
    if not absolut_path.exists():
        absolut_path.mkdir(parents=True, exist_ok=True)
    
    return absolut_path


def get_log_file_path():
    lib_path = get_library_path()
    log_filename = "runtime.log"
    return lib_path / log_filename
