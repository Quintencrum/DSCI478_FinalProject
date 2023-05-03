import matplotlib.pyplot as plt
import pandas as pd

from data.dataClass import Data, DataType, IBBData


def plot(data: Data):
    fig = plt.figure(data.get_plot())

    plt.show()

def time_series(data: Data):
    if data == DataType.IBB:
        return _time_series_ibb(data)
    elif data == DataType.ETFS:
        return _time_series_etfs(data)
    elif data == DataType.SPX:
        return _time_series_spx(data)

def _time_series_ibb(data: IBBData):
    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs[0,0].plot(data.df)
    axs[0,1].plot(data.lp_data.df)
    axs[1,0].plot(data.nc_data.df)
    axs[1,1].plot(data.pc_data.df)

    plt.show()

def _time_series_etfs(data: Data):
    pass

def _time_series_spx(data: Data):
    pass

