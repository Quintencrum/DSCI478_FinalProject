import argparse

from data import dataImport, visualize
from models import networkGraphs_dataPrep, networkGraphs_plot
import numpy as np


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

    close_dist_matrix = networkGraphs_dataPrep.df_distance_correlation(allStock_df_list[0])
    # open_dist_matrix = networkGraphs_dataPrep.df_distance_correlation(allStock_df_list[1])
    # open_close_dist_matrix = networkGraphs_dataPrep.df_distance_correlation(allStock_df_list[2])
    # high_dist_matrix = networkGraphs_dataPrep.df_distance_correlation(allStock_df_list[3])
    # low_dist_matrix = networkGraphs_dataPrep.df_distance_correlation(allStock_df_list[4])

    corrNet_close = networkGraphs_dataPrep.build_corr(close_dist_matrix)
    # corrNet_open = networkGraphs_dataPrep.build_corr(open_dist_matrix)
    # corrNet_close_diff = networkGraphs_dataPrep.build_corr(open_close_dist_matrix)
    # corrNet_high = networkGraphs_dataPrep.build_corr(high_dist_matrix)
    # corrNet_low = networkGraphs_dataPrep.build_corr(low_dist_matrix)

    networkGraphs_plot.plot_network(corrNet_close,'Hello')






    # df_dcor_list = [networkGraphs_dataPrep.distance_correlation_matrix(df) for df in allStock_df_list]

    # df_dcor_list = [networkGraphs_dataPrep.df_distance_correlation(df) for df in allStock_df_list]

    # print(df_dcor_list[4].head())


    # dist_corr_matrix = networkGraphs_dataPrep.distance_correlation_matrix(allStock_df_list[0])
    # print(allStock_df_list[0])
    # print(dist_corr_matrix)

    # networkGraphs_plot.plot_network(df_dcor_list[4],'Hello')



if __name__ == "__main__":
    main()