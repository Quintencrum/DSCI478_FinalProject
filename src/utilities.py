from pathlib import Path


def get_project_path():
    return Path.cwd()

def get_data_path():
    return get_project_path().parent.joinpath("data")
