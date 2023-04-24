import argparse

from data import dataImport, visualize


def main(parsed):
    if parsed.visulize:
        print("T1")
    print("T2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Main file for the project",
        epilog="Some of the options will generate local artifacts")

    parser.add_argument('-v', '--visulize', action='store_true')

    main(parser.parse_args())
