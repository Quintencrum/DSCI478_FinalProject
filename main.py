import argparse

from data import dataImport, visualize
from models import networkGraphsFull, networkGraphs_temp
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
    
    # #all stocks data
    allStock_df_list = networkGraphsFull.networkDataPrep(dataImport.data_import_ng())

    # #distance matrices
    # close_dist_matrix = networkGraphsFull.df_distance_correlation(allStock_df_list[0])
    # open_dist_matrix = networkGraphsFull.df_distance_correlation(allStock_df_list[1])
    # open_close_dist_matrix = networkGraphsFull.df_distance_correlation(allStock_df_list[2])
    # high_dist_matrix = networkGraphsFull.df_distance_correlation(allStock_df_list[3])
    # low_dist_matrix = networkGraphsFull.df_distance_correlation(allStock_df_list[4])

    # #correlationNetworks
    # corrNet_close = networkGraphsFull.build_corr(close_dist_matrix)
    # corrNet_open = networkGraphsFull.build_corr(open_dist_matrix)
    # corrNet_close_diff = networkGraphsFull.build_corr(open_close_dist_matrix)
    # corrNet_high = networkGraphsFull.build_corr(high_dist_matrix)
    # corrNet_low = networkGraphsFull.build_corr(low_dist_matrix)

    # #plotting the various network graphs
    # networkGraphsFull.plot_network(corrNet_close,'Distance Correlation Network for Closing Prices')
    # networkGraphsFull.plot_network(corrNet_open,'Distance Correlation Network for Opening Prices')
    # networkGraphsFull.plot_network(corrNet_close_diff,'Distance Correlation Network for Difference in Prices (Open vs Close)')
    # networkGraphsFull.plot_network(corrNet_high,'Distance Correlation Network for High Prices')
    # networkGraphsFull.plot_network(corrNet_low,'Distance Correlation Network for Low Prices')

    # networkGraphsFull.networkGraphsMainFunction(allStock_df_list)

    # allStock_df_list_inner, allStock_df_list_outer, allStock_df_list = networkGraphsFull.subsetStocks(allStock_df_list)

    corrNet_close, corrNet_open, corrNet_close_diff, corrNet_high, corrNet_low = networkGraphsFull.correlation_networks(allStock_df_list)

    # networkGraphsFull.risk_graph(corrNet_close,'close_combined')
    # networkGraphsFull.risk_graph(corrNet_open,'open_combined')
    # networkGraphsFull.risk_graph(corrNet_close_diff,'difference_combined')
    # networkGraphsFull.risk_graph(corrNet_high,'high_combined')
    # networkGraphsFull.risk_graph(corrNet_low,'low_combined')
    networkGraphsFull.run_risk_graphs(allStock_df_list)
    





if __name__ == "__main__":
    main()
