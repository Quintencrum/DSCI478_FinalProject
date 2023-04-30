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


def distCorr(x, y):
    n = len(x)
    a = np.zeros((n, n))
    b = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            a[i, j] = abs(x[i] - x[j])
            a[j, i] = a[i, j]
            b[i, j] = abs(y[i] - y[j])
            b[j, i] = b[i, j]

    C = 0.0
    D = 0.0
    E = 0.0
    F = 0.0
    for i in range(n):
        C += np.sum(a[i, :])
        D += np.sum(b[i, :])
        E += np.sum(a[i, :]) ** 2
        F += np.sum(b[i, :]) ** 2

    covXY = 0.0
    for i in range(n):
        for j in range(n):
            covXY += a[i, j] * b[i, j]

    dcovXY = covXY - C * D / n
    dvarX = (E - C ** 2 / n) / n
    dvarY = (F - D ** 2 / n) / n

    if dvarX * dvarY == 0:
        return 0.0
    else:
        return np.sqrt(dcovXY / np.sqrt(dvarX * dvarY))





def distance_correlation_matrix(df):
    df_dcor = pd.DataFrame(index=df.columns, columns=df.columns)
    
    # Pairwise Euclidean distance matrix
    dist_mat = np.sqrt((df ** 2).sum(axis=0).values.reshape(-1,1) + 
                       (df ** 2).sum(axis=0).values - 2 * df.T.dot(df))
    
    #                   = distance mat - row mean - column mean + total mean
    # dist_mat_centered = dist_mat - dist_mat.mean(axis=1).reshape(-1,1) - dist_mat.mean(axis=0) + dist_mat.mean()
    row_mean = dist_mat.mean(axis=1).values
    column_mean = dist_mat.mean(axis=0)
    dist_mat_centered = dist_mat - row_mean.reshape(-1,1) - column_mean + dist_mat.mean()

    
    # hopefully compute the sample covariance and variances of the row and column vectors
    cov_mat = np.cov(dist_mat_centered, rowvar=False)
    row_vars = np.var(dist_mat_centered, axis=1, ddof=1)
    col_vars = np.var(dist_mat_centered, axis=0, ddof=1)
    
    # computes distance correlation coefficients
    for i, stock_i in enumerate(df.columns):
        for j, stock_j in enumerate(df.columns):
            numerator = cov_mat[i, j]
            denominator = np.sqrt(row_vars[i] * col_vars[j])
            if denominator == 0:
                dcor_val = 0
            else:
                dcor_val = numerator / denominator

            df_dcor.at[stock_i, stock_j] = dcor_val

    np.fill_diagonal(df_dcor.values, 1)
    
    return df_dcor


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
            distcorr_ij = distCorr(stock_i, stock_j)
            distcorr_matrix[i, j] = distcorr_ij
            distcorr_matrix[j, i] = distcorr_ij

    distcorr_df = pd.DataFrame(distcorr_matrix, columns=df_train.columns, index=df_train.columns)
    return distcorr_df



