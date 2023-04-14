import numpy as np
import pandas as pd
import matplotlib as plt
import sklearn as sklearn
import tensorflow as tf



def data_Import_ibb():
    ibb_full = pd.read_csv('data/Deep_Portfolio/ibb.csv', index_col=0).astype('float32')

    stock_lp = pd.read_csv('data/Deep_Portfolio/last_price.csv', index_col=0).dropna(axis=1, how='any').astype('float32')
    stock_calibrate_lp = stock_lp.iloc[0:104, :]
    stock_validate_lp = stock_lp.iloc[104:, :]

    stock_net = pd.read_csv('data/Deep_Portfolio/net_change.csv', index_col=0).dropna(axis=1, how='any').astype('float32')
    stock_calibrate_net = stock_net.iloc[0:104, :]
    stock_validate_net = stock_net.iloc[104:, :]

    stock_percentage = pd.read_csv('data/Deep_Portfolio/percentage_change.csv', index_col=0).dropna(axis=1, how='any').astype('float32')
    stock_calibrate_percentage = stock_percentage.iloc[0:104, :]
    stock_validate_percentage = stock_percentage.iloc[104:, :]

    return ibb_full, stock_lp, stock_calibrate_lp, stock_validate_lp, stock_net, stock_calibrate_net, stock_validate_net, stock_percentage, stock_calibrate_percentage, stock_validate_percentage


def data_import_ETFs():
    

# def data_import_spx():
#     pd.read_csv('data/PCA_Factors/spx_holdings_and_spx_closeprice.csv')