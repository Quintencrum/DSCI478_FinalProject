import argparse

from data import dataImport, visualize
from data.dataClass import DataType


def main(parsed):
    if parsed.visualize:
        visu_data(parsed)

def visu_data(parsed):
    data = dataImport.get_data(parsed.visualize)
    visualize.plot(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Main file for the project",
        epilog="Some of the options will generate local artifacts")

    parser.add_argument('-v', '--visualize', choices=list(DataType), type=DataType)

    parser.add_argument('-t')

    main(parser.parse_args())
