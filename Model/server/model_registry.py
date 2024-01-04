import os
from pathlib import Path
import shutil

#All done by: Shahd

def get_latest_model_version():
    model_dir = 'Model/model_registry' 
    model_files = Path(model_dir).glob('model_version_*.h5')
    
    sorted_models = sorted(model_files, key=lambda f: os.path.getmtime(f), reverse=True)
    
    if sorted_models:
        latest_version = sorted_models[0]
        print(str(latest_version))
        return str(latest_version)
    else:
        return None  # No models found

def get_model_by_version(version):
    model_dir = 'Model/model_registry' 
    model_file = list(Path(model_dir).glob(f'model_version_{version}.h5'))
    
    if model_file:
        print(str(model_file[0]))
        return str(model_file[0])
    else:
        return None  # No models found

def set_active_model(version):
    active_model_path = get_model_by_version(version)

    if active_model_path:
        # Copy the selected model to a temporary location
        model_dir = 'Model/model_registry' 
        temp_model_path = Path(model_dir) / f"temp_{version}.h5"
        shutil.copy(active_model_path, temp_model_path)

        os.remove(active_model_path)

        shutil.move(temp_model_path, active_model_path)

        os.utime(active_model_path)

        active_model_path = active_model_path
        return active_model_path
    
def get_all_models():
    model_dir = 'Model/model_registry' 
    model_files = list(Path(model_dir).glob('model_version_*.h5'))
    retrained_files = list(Path(model_dir).glob('retrained_model_version_*.h5'))
    
    if model_files or retrained_files:
        version_numbers = [str(model_file.stem) for model_file in model_files + retrained_files]
        for version_number in version_numbers:
            print(version_number)
        return version_numbers
    else:
        return None  # No models found
    

