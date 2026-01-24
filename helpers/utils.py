import os 


def get_filename(file_path: str) -> str:
    return os.path.basename(file_path)
