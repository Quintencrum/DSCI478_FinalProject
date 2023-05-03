import pandas as pd

from data.dataClass import Data, DataType, IBBData
import utilities as utils


def data_import_ibb():
    data_path = utils.get_data_path().joinpath('Deep_Portfolio')
    
    ibb_full_df = pd.read_csv(data_path.joinpath('ibb.csv'), index_col=0).astype('float32')
    stock_lp_df = pd.read_csv(data_path.joinpath('last_price.csv'), index_col=0).astype('float32')
    stock_net_df = pd.read_csv(data_path.joinpath('net_change.csv'), index_col=0).astype('float32')
    stock_perc_df = pd.read_csv(data_path.joinpath('percentage_change.csv'), index_col=0).astype('float32')

    ibb_full = IBBData(df=ibb_full_df, df_lp=stock_lp_df, df_nc=stock_net_df, df_pc=stock_perc_df,
                       name="IBB Index Data")

    return ibb_full


def data_import_etfs():
    data_path = utils.get_data_path().joinpath('OLPS_Portfolio')

    etfs_df = pd.read_csv(data_path.joinpath('ETFs.csv'), index_col=0).astype('float32')
    etfs = Data(etfs_df, DataType.ETFS, "List of ETFs")
    
    return etfs

def data_import_spx():
    data_path = utils.get_data_path().joinpath('PCA_Factors')

    spx_df = pd.read_csv(data_path.joinpath('spx_holdings_and_spx_closeprice.csv'), index_col=0).astype('float32')
    spx = Data(spx_df, DataType.SPX, "SPX Index")

    return spx

def data_import_ng():
    data_path = utils.get_data_path().joinpath('Network_graphs')

    # ng_df = pd.read_csv(data_path.joinpath('all_stocks.csv'), index_col=0).astype('float32')
    ng_df = pd.read_csv(data_path.joinpath('all_stocks.csv'))
    #index is set in networkGraphs_dataPrep.py to make them dateTimes not just dates
    ng = Data(ng_df, DataType.NG, "All Stocks Simple Data")

    return ng

def get_data(dType: DataType):
    if dType == DataType.IBB:
        return data_import_ibb()
    elif dType == DataType.ETFS:
        return data_import_etfs()
    elif dType == DataType.SPX:
        return data_import_spx()
    elif dType == DataType.NG:
        return data_import_ng()
