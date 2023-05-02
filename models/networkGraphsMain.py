# import sys
# sys.path.append('../data')

# from data import dataImport, visualize
# from models import networkGraphs_dataPrep, networkGraphs_plot, networkGraphsFull
# import numpy as np

from networkGraphsFull import plot_network, df_distance_correlation, build_corr


def plot_network_graphs(corrNet_close, corrNet_open, corrNet_close_diff, corrNet_high, corrNet_low):
    #plotting the various network graphs
    plot_network(corrNet_close,'Distance Correlation Network for Closing Prices')
    plot_network(corrNet_open,'Distance Correlation Network for Opening Prices')
    plot_network(corrNet_close_diff,'Distance Correlation Network for Difference in Prices (Open vs Close)')
    plot_network(corrNet_high,'Distance Correlation Network for High Prices')
    plot_network(corrNet_low,'Distance Correlation Network for Low Prices')
    return

def correlation_networks(allStock_df_list):
    #distance matrices
    close_dist_matrix = df_distance_correlation(allStock_df_list[0])
    open_dist_matrix = df_distance_correlation(allStock_df_list[1])
    open_close_dist_matrix = df_distance_correlation(allStock_df_list[2])
    high_dist_matrix = df_distance_correlation(allStock_df_list[3])
    low_dist_matrix = df_distance_correlation(allStock_df_list[4])

    #correlationNetworks
    corrNet_close = build_corr(close_dist_matrix)
    corrNet_open = build_corr(open_dist_matrix)
    corrNet_close_diff = build_corr(open_close_dist_matrix)
    corrNet_high = build_corr(high_dist_matrix)
    corrNet_low = build_corr(low_dist_matrix)

    return [close_dist_matrix, open_dist_matrix, open_close_dist_matrix, high_dist_matrix, low_dist_matrix]





def full_stock_sets(allStock_df_list):

    corrNet_close, corrNet_open, corrNet_close_diff, corrNet_high, corrNet_low = correlation_networks(allStock_df_list)

    plot_network_graphs(corrNet_close, corrNet_open, corrNet_close_diff, corrNet_high, corrNet_low)



def select_stock_set(allStock_df_list):
    #stocks to keep for different categories

    #closing
    closing_subset_outer = ['Apple','Amazon','Altaba','United Health','Walmart']
    closing_subset_inner = ['United Technologies','3M','JPMorgan Chase','IBM','American Express']
    closing_subset = closing_subset_inner + closing_subset_outer

    #open
    opening_subset_outer = ['Apple','Amazon','Altaba','United Health','Home Depot']
    opening_subset_inner = ['United Technologies','3M','JPMorgan Chase','Cisco Systems','American Express']
    opening_subset = opening_subset_inner + opening_subset_outer

    #open-close
    diff_subset_outer = ['Apple','Amazon','Altaba','United Health','Nike']
    diff_subset_inner = ['United Technologies','3M','General Electric','IBM','American Express']
    diff_subset = diff_subset_inner + diff_subset_outer

    #high
    high_subset_outer = ['Procter & Gamble','Walmart','Merk','United Health','Verizon']
    high_subset_inner = ['JPMorgan Chase','3M','General Electric','IBM','American Express']
    high_subset = high_subset_inner + high_subset_outer

    #low
    low_subset_outer = ['Apple','Amazon','Altaba','Verizon','Procter & Gamble']
    low_subset_inner = ['United Technologies','3M','General Electric','Walt Disney','American Express']
    low_subset = low_subset_inner + low_subset_outer


    print(allStock_df_list)

    # corrNet_close, corrNet_open, corrNet_close_diff, corrNet_high, corrNet_low = correlation_networks(allStock_df_list)

    # #plotting the various network graphs
    # plot_network_graphs(corrNet_close, corrNet_open, corrNet_close_diff, corrNet_high, corrNet_low)




def networkGraphsMainFunction(allStock_df_list):
    # full_stock_sets(allStock_df_list)
    select_stock_set(allStock_df_list)