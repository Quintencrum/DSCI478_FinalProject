import argparse
import numpy as np

from data import dataImport, visualize
from data.dataClass import DataType
from models.modelsClass import ModelType
    

def main(parsed: argparse.Namespace):
    if parsed.visualize:
        visu_data(parsed)
    elif parsed.model:
       run_model(parsed)

def run_model(parsed: argparse.Namespace):
    if parsed.model == ModelType.NG:
        run_network_graphs()
    elif parsed.model == ModelType.LN:
        run_lin_reg()

def run_lin_reg():
    from models import LinReg_Evan

    LinReg_Evan.main()

def run_network_graphs():
    from models import networkGraphsFull, networkGraphs_dataPrep, networkGraphs_plot

    # #all stocks data
    allStock_data = dataImport.data_import_ng()
    allStock_df_list = networkGraphsFull.networkDataPrep(allStock_data.df)

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

    networkGraphsFull.networkGraphsMainFunction(allStock_df_list)

def visu_data(parsed: argparse.Namespace):
    data = dataImport.get_data(parsed.visualize)
    visualize.plot(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Main file for the project",
        epilog="Some of the options will generate local artifacts")

    parser.add_argument('-v', '--visualize', choices=list(DataType), type=DataType)

    parser.add_argument('-m', '--model', choices=list(ModelType), type=ModelType)

    main(parser.parse_args())
