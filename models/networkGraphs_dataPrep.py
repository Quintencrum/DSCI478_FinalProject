import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
import tensorflow as tf


# all_stocks = data_import_ng()
# all_stocks.Name.unique()


def networkDataPrep(allStock_df):
    # print('OG dataframe: ')
    # print('head--')
    # print(allStock_df.head())
    # print('info--')
    # print(allStock_df.info)
    # print('unique tickers--')
    # print(allStock_df['Name'].unique())
    # print('End of OG dataframe info')

    #A little bit of data updates to allow for better plotting and modeling

    #Starting with adding in real names:
    tickers_names = {'AABA':'Altaba', 
                  'AAPL':'Apple', 
                  'AMZN': 'Amazon',
                  'AXP':'American Express',
                  'BA':'Boeing', 
                  'CAT':'Caterpillar',
                  'MMM':'3M', 
                  'CVX':'Chevron', 
                  'CSCO':'Cisco Systems',
                  'KO':'Coca-Cola', 
                  'DIS':'Walt Disney', 
                  'XOM':'Exxon Mobil',
                  'GE': 'General Electric',
                  'GS':'Goldman Sachs',
                  'HD': 'Home Depot',
                  'IBM': 'IBM',
                  'INTC': 'Intel',
                  'JNJ':'Johnson & Johnson',
                  'JPM':'JPMorgan Chase',
                  'MCD':'Mcdonald\'s',
                  'MRK':'Merk',
                  'MSFT':'Microsoft',
                  'NKE':'Nike',
                  'PFE':'Pfizer',
                  'PG':'Procter & Gamble',
                  'TRV':'Travelers',
                  'UTX':'United Technologies',
                  'UNH':'UnitedHealth',
                  'VZ':'Verizon',
                  'WMT':'Walmart',
                  'GOOGL':'Google'}

    allStock_df['Name_full'] = allStock_df['Name'].map(tickers_names)
    
    #updating dates
    allStock_df['Date'] = pd.to_datetime(allStock_df['Date'])
    #set dates as index (mentioned in dataImport.py)
    allStock_df.set_index('Date',inplace=True)

    #adding in close difference price
    allStock_df['OpenClose_Diff'] = allStock_df.loc[:,'Close'] - allStock_df.loc[:,'Open']

    df_close = allStock_df.pivot(columns='Name_full', values='Close')
    df_open = allStock_df.pivot(columns='Name_full', values='Open')
    df_open_close_diff = allStock_df.pivot(columns='Name_full', values='OpenClose_Diff')
    df_high = allStock_df.pivot(columns='Name_full', values='High')
    df_low = allStock_df.pivot(columns='Name_full', values='Low')

    # print(df_close.head())
    # print(df_open.head())
    # print(df_open_close_diff.head())
    # print(df_high.head())
    # print(df_low.head())
    
    allStock_df_list = [df_close, df_open,df_open_close_diff, df_high, df_low]

    #detrends each time series for each df
    stocks = df_close.columns.tolist()
    for i in allStock_df_list:
        for j in stocks:
            i[j] = i[j].diff()

    #drops all missing values in each DataFrame
    for i in allStock_df_list:
        i.dropna(inplace=True)

    return allStock_df_list



def distCorr_optimized(x, y):
    n = len(x)
    a = np.abs(np.subtract.outer(x, x))
    b = np.abs(np.subtract.outer(y, y))
    A = a - a.mean(axis=0) - a.mean(axis=1)[:, np.newaxis] + a.mean()
    B = b - b.mean(axis=0) - b.mean(axis=1)[:, np.newaxis] + b.mean()
    dcov = np.sqrt(np.sum(A * B)) / n
    dcov_xx = np.sqrt(np.sum(A * A)) / n
    dcov_yy = np.sqrt(np.sum(B * B)) / n
    dcor = 0 if dcov_xx == 0 or dcov_yy == 0 else dcov / np.sqrt(dcov_xx * dcov_yy)
    return dcor



def df_distance_correlation(df_train):
    print('start')
    n_stocks = len(df_train.columns)
    distcorr_matrix = np.zeros((n_stocks, n_stocks))
    # print(n_stocks) #31 :(
    for i in range(n_stocks):
        print(i)
        for j in range(i, n_stocks):
            print(j)
            stock_i = df_train.iloc[:, i].values
            stock_j = df_train.iloc[:, j].values
            # distcorr_ij = distCorr(stock_i, stock_j)
            distcorr_ij = distCorr_optimized(stock_i, stock_j)
            distcorr_matrix[i, j] = distcorr_ij
            distcorr_matrix[j, i] = distcorr_ij

    distcorr_df = pd.DataFrame(distcorr_matrix, columns=df_train.columns, index=df_train.columns)
    return distcorr_df

