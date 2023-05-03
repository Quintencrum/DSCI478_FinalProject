from pathlib import Path


def get_project_path():
    return Path(__file__).parent

def get_data_path():
    return get_project_path().joinpath("data")
