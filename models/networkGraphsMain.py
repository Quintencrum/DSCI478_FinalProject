from data import dataImport, visualize
from models import networkGraphs_dataPrep, networkGraphs_plot, networkGraphsFull
import numpy as np


def plot_network_graphs(corrNet_close, corrNet_open, corrNet_close_diff, corrNet_high, corrNet_low):
    #plotting the various network graphs
    networkGraphsFull.plot_network(corrNet_close,'Distance Correlation Network for Closing Prices')
    networkGraphsFull.plot_network(corrNet_open,'Distance Correlation Network for Opening Prices')
    networkGraphsFull.plot_network(corrNet_close_diff,'Distance Correlation Network for Difference in Prices (Open vs Close)')
    networkGraphsFull.plot_network(corrNet_high,'Distance Correlation Network for High Prices')
    networkGraphsFull.plot_network(corrNet_low,'Distance Correlation Network for Low Prices')
    return

def correlation_networks(allStock_df_list):
    #distance matrices
    close_dist_matrix = networkGraphsFull.df_distance_correlation(allStock_df_list[0])
    open_dist_matrix = networkGraphsFull.df_distance_correlation(allStock_df_list[1])
    open_close_dist_matrix = networkGraphsFull.df_distance_correlation(allStock_df_list[2])
    high_dist_matrix = networkGraphsFull.df_distance_correlation(allStock_df_list[3])
    low_dist_matrix = networkGraphsFull.df_distance_correlation(allStock_df_list[4])

    #correlationNetworks
    corrNet_close = networkGraphsFull.build_corr(close_dist_matrix)
    corrNet_open = networkGraphsFull.build_corr(open_dist_matrix)
    corrNet_close_diff = networkGraphsFull.build_corr(open_close_dist_matrix)
    corrNet_high = networkGraphsFull.build_corr(high_dist_matrix)
    corrNet_low = networkGraphsFull.build_corr(low_dist_matrix)

    return [close_dist_matrix, open_dist_matrix, open_close_dist_matrix, high_dist_matrix, low_dist_matrix]





def full_stock_sets():
    #all stocks data
    allStock_df_list = networkGraphsFull.networkDataPrep(dataImport.data_import_ng())
    corrNet_close, corrNet_open, corrNet_close_diff, corrNet_high, corrNet_low = correlation_networks(allStock_df_list)
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
    plot_network_graphs(corrNet_close, corrNet_open, corrNet_close_diff, corrNet_high, corrNet_low)
    # networkGraphsFull.plot_network(corrNet_close,'Distance Correlation Network for Closing Prices')
    # networkGraphsFull.plot_network(corrNet_open,'Distance Correlation Network for Opening Prices')
    # networkGraphsFull.plot_network(corrNet_close_diff,'Distance Correlation Network for Difference in Prices (Open vs Close)')
    # networkGraphsFull.plot_network(corrNet_high,'Distance Correlation Network for High Prices')
    # networkGraphsFull.plot_network(corrNet_low,'Distance Correlation Network for Low Prices')


def select_stock_set():

    #stocks to keep for different categories

    #closing
    closing_subset_outer = ['Apple','Amazon','Altaba','United Health','Walmart']
    closing_subset_inner = ['United Technologies','3M','JPMorgan Chase','IBM','American Express']
    closing_subset = closing_subset_inner + closing_subset_outer

    #open
    opening_subset_outer = ['Apple','Amazon','Altaba','United Health','Home Depot']
    opening_subset_inner = ['United Technologies','3M','JPMorgan Chase','Cisco Systems','American Express']
    opening_subset = opening_subset_inner + opening_subset_outer






full_stock_sets()