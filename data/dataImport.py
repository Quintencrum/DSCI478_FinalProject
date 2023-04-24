import pandas as pd

import utilities as utils


def data_import_ibb():
    data_path = utils.get_data_path().joinpath('Deep_Portfolio')
    
    ibb_full = pd.read_csv(data_path.joinpath('ibb.csv'), index_col=0).astype('float32')
    stock_lp = pd.read_csv(data_path.joinpath('last_price.csv'), index_col=0).astype('float32')
    stock_net = pd.read_csv(data_path.joinpath('net_change.csv'), index_col=0).astype('float32')
    stock_percentage = pd.read_csv(data_path.joinpath('percentage_change.csv'), index_col=0).astype('float32')

    return ibb_full, stock_lp, stock_net, stock_percentage


def data_import_etfs():
    data_path = utils.get_data_path().joinpath('OLPS_Portfolio')

    etfs_full = pd.read_csv(data_path.joinpath('ETFs.csv'), index_col=0).astype('float32')

    return etfs_full

def data_import_spx():
    data_path = utils.get_data_path().joinpath('PCA_Factors')

    spx_full = pd.read_csv(data_path.joinpath('spx_holdings_and_spx_closeprice.csv'), index_col=0).astype('float32')

    return spx_full
