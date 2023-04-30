import argparse

from data import dataImport, visualize
from models import networkGraphs_dataPrep, networkGraphs_plot


# def main(parsed):
#     if parsed.visulize:
#         print("T1")
#     print("T2")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Main file for the project",
#         epilog="Some of the options will generate local artifacts")

#     parser.add_argument('-v', '--visulize', action='store_true')

#     main(parser.parse_args())

def main():
    print('Hello World!')
    allStock_df_list = networkGraphs_dataPrep.networkDataPrep(dataImport.data_import_ng())

    df_dcor_list = [networkGraphs_dataPrep.distance_correlation_matrix(df) for df in allStock_df_list]

    print(df_dcor_list[4].head())

    # dist_corr_matrix = networkGraphs_dataPrep.distance_correlation_matrix(allStock_df)

    # print(dist_corr_matrix)



if __name__ == "__main__":
    main()
