import numpy as np
import pandas as pd
import matplotlib as plt
import sklearn as sklearn
import tensorflow as tf



def data_Import():
    stock = [[]]

    ibb_full = pd.read_csv('data/Deep_Portfolio/ibb.csv', index_col=0).astype('float32')

    stock_lp = pd.read_csv('data/Deep_Portfolio/last_price.csv', index_col=0).dropna(axis=1, how='any').astype('float32')
    # stock['calibrate']['lp'] = stock_lp.iloc[0:104, :]
    # stock['validate']['lp'] = stock_lp.iloc[104:, :]

    stock_net = pd.read_csv('data/Deep_Portfolio/net_change.csv', index_col=0).dropna(axis=1, how='any').astype('float32')
    # stock['calibrate']['net'] = stock_net.iloc[0:104, :]
    # stock['validate']['net'] = stock_net.iloc[104:, :]

    stock_percentage = pd.read_csv('data/Deep_Portfolio/percentage_change.csv', index_col=0).dropna(axis=1, how='any').astype('float32')
    # stock['calibrate']['percentage'] = stock_percentage.iloc[0:104, :]
    # stock['validate']['percentage'] = stock_percentage.iloc[104:, :]

    return ibb_full, stock, stock_lp, stock_net, stock_percentage


data_Import()