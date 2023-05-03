import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
import tensorflow as tf
import networkx as nx


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
                  'UNH':'United Health',
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
    cov = np.sqrt(np.sum(A * B)) / n
    cov_xx = np.sqrt(np.sum(A * A)) / n
    cov_yy = np.sqrt(np.sum(B * B)) / n
    dcor = 0 if cov_xx == 0 or cov_yy == 0 else cov / np.sqrt(cov_xx * cov_yy)
    return dcor



def df_distance_correlation(df_train):
    print('start')
    n_stocks = len(df_train.columns)
    distcorr_matrix = np.zeros((n_stocks, n_stocks))
    # print(n_stocks) #31 :(
    for i in range(n_stocks):
        # print(i)
        for j in range(i, n_stocks):
            # print(j)
            stock_i = df_train.iloc[:, i].values
            stock_j = df_train.iloc[:, j].values
            # distcorr_ij = distCorr(stock_i, stock_j)
            distcorr_ij = distCorr_optimized(stock_i, stock_j)
            distcorr_matrix[i, j] = distcorr_ij
            distcorr_matrix[j, i] = distcorr_ij

    distcorr_df = pd.DataFrame(distcorr_matrix, columns=df_train.columns, index=df_train.columns)
    return distcorr_df



def build_corr(df_train):
    cor_matrix = df_train.values.astype('float')
    
    # temp_mat_1 = nx.from_numpy_matrix(1 - cor_matrix)
    temp_mat_1 = nx.from_numpy_array(1 - cor_matrix)


    stock_names = df_train.index.values
    
    # rename the nodes with the stock names
    temp_mat_1 = nx.relabel_nodes(temp_mat_1, lambda x: stock_names[x])
    
    temp_mat_1.edges(data=True)

    temp_mat_2 = temp_mat_1.copy()
    
    for (u, v, wt) in temp_mat_1.edges.data('weight'):
        if wt >= 1 - 0.325:
            temp_mat_2.remove_edge(u, v)
        if u == v:
            temp_mat_2.remove_edge(u, v)
    
    return temp_mat_2





def plot_network(temp, title):
    # print(type(temp))
    # print('-----------------------?????????????')
    edges, weights = zip(*nx.get_edge_attributes(temp, "weight").items())

    pos = nx.kamada_kawai_layout(temp) #not sure exactly what this function is but I found it on the web and it works!!!!!!

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title(title, size=16)

    deg = temp.degree
    nodelist = []
    node_sizes = []

    for n, d in deg:    #helps with names
        nodelist.append(n)
        node_sizes.append(d)

    nx.draw_networkx_nodes(
        temp,
        pos,
        node_color="#DA70D6",
        nodelist=nodelist,
        node_size=np.power(node_sizes, 2.33),
        alpha=0.8,
        ax=ax,
    )

    nx.draw_networkx_labels(temp, pos, font_size=13, font_family="sans-serif", ax=ax)

    cmap = plt.cm.cubehelix_r

    nx.draw_networkx_edges(
        temp,
        pos,
        edgelist=edges,
        style="solid",
        edge_color=weights,
        edge_cmap=cmap,
        edge_vmin=min(weights),
        edge_vmax=max(weights),
        ax=ax,
    )
#colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap, 
        norm=plt.Normalize(vmin=min(weights), 
        vmax=max(weights))
    )
    sm._A = []
    plt.colorbar(sm)
    plt.axis("off")

    # #silence warnings   
    # import warnings
    # warnings.filterwarnings("ignore")
    outputStringName = 'output_' + str(title) + '.jpg'
    plt.savefig(outputStringName)


def risk_graph(corrNet, inputName):
    comm_bet_cent = nx.communicability_betweenness_centrality(corrNet)

    risk_alloc1 = pd.Series(comm_bet_cent)
    risk_alloc2 = risk_alloc1 / risk_alloc1.sum()
    risk_alloc3 = risk_alloc2.sort_values()

    #plotting time
    plt.figure(figsize=(8, 10))
    plt.barh(y=risk_alloc3.index, width=risk_alloc3.values, color='r')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.title("Intraportfolio Risk " + str(inputName), size=18)

    for i, (stock, risk) in enumerate(zip(risk_alloc3.index, risk_alloc3.values)):
        plt.annotate(f"{(risk * 100):.2f}%", xy=(risk + 0.001, i + 0.15), fontsize=10)

    plt.xticks([])
    plt.xlabel("Relative Risk %", size=12)
    # plt.show()
    outputStringName = 'relativeRisk_' + str(inputName)
    plt.savefig(outputStringName)


def plot_network_graphs(corrNet_close, corrNet_open, corrNet_close_diff, corrNet_high, corrNet_low,type):
    #plotting the various network graphs
    plot_network(corrNet_close,'Distance Correlation Network for Closing Prices-'+str(type))
    plot_network(corrNet_open,'Distance Correlation Network for Opening Prices-'+str(type))
    plot_network(corrNet_close_diff,'Distance Correlation Network for Difference in Prices (Open vs Close)-'+str(type))
    plot_network(corrNet_high,'Distance Correlation Network for High Prices-'+str(type))
    plot_network(corrNet_low,'Distance Correlation Network for Low Prices-'+str(type))
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

    return [corrNet_close, corrNet_open, corrNet_close_diff, corrNet_high, corrNet_low]





def full_stock_sets(allStock_df_list):

    corrNet_close, corrNet_open, corrNet_close_diff, corrNet_high, corrNet_low = correlation_networks(allStock_df_list)

    plot_network_graphs(corrNet_close, corrNet_open, corrNet_close_diff, corrNet_high, corrNet_low,'full')

def subsetStocks(allStock_df_list):
    #closing
    closing_subset_outer = ['Apple','Amazon','Altaba','United Health','Walmart']
    closing_subset_inner = ['United Technologies','3M','JPMorgan Chase','IBM','American Express']
    closing_subset = closing_subset_inner + closing_subset_outer

    df_close_outer = allStock_df_list[0][closing_subset_outer]
    df_close_inner = allStock_df_list[0][closing_subset_inner]
    df_close = allStock_df_list[0][closing_subset]

    #open
    opening_subset_outer = ['Apple','Amazon','Altaba','United Health','Home Depot']
    opening_subset_inner = ['United Technologies','3M','JPMorgan Chase','Cisco Systems','American Express']
    opening_subset = opening_subset_inner + opening_subset_outer

    df_open_outer = allStock_df_list[0][opening_subset_outer]
    df_open_inner = allStock_df_list[0][opening_subset_inner]
    df_open = allStock_df_list[0][opening_subset]

    #open-close
    diff_subset_outer = ['Apple','Amazon','Altaba','United Health','Nike']
    diff_subset_inner = ['United Technologies','3M','General Electric','IBM','American Express']
    diff_subset = diff_subset_inner + diff_subset_outer

    df_diff_outer = allStock_df_list[0][diff_subset_outer]
    df_diff_inner = allStock_df_list[0][diff_subset_inner]
    df_diff = allStock_df_list[0][diff_subset]

    #high
    high_subset_outer = ['Procter & Gamble','Walmart','Merk','United Health','Verizon']
    high_subset_inner = ['JPMorgan Chase','3M','General Electric','IBM','American Express']
    high_subset = high_subset_inner + high_subset_outer

    df_high_outer = allStock_df_list[0][high_subset_outer]
    df_high_inner = allStock_df_list[0][high_subset_inner]
    df_high = allStock_df_list[0][high_subset]

    #low
    low_subset_outer = ['Apple','Amazon','Altaba','Verizon','Procter & Gamble']
    low_subset_inner = ['United Technologies','3M','General Electric','Walt Disney','American Express']
    low_subset = low_subset_inner + low_subset_outer

    df_low_outer = allStock_df_list[0][low_subset_outer]
    df_low_inner = allStock_df_list[0][low_subset_inner]
    df_low = allStock_df_list[0][low_subset]


    #combining stocks
    allStock_df_list_inner = [df_close_inner, df_open_inner,df_diff_inner, df_high_inner, df_low_inner]
    allStock_df_list_outer = [df_close_outer, df_open_outer,df_diff_outer, df_high_outer, df_low_outer]
    allStock_df_list = [df_close, df_open,df_diff, df_high, df_low]

    return [allStock_df_list_inner, allStock_df_list_outer, allStock_df_list]



def select_stock_set(allStock_df_list):
    #stocks to keep for different categories

    # #closing
    # closing_subset_outer = ['Apple','Amazon','Altaba','United Health','Walmart']
    # closing_subset_inner = ['United Technologies','3M','JPMorgan Chase','IBM','American Express']
    # closing_subset = closing_subset_inner + closing_subset_outer

    # df_close_outer = allStock_df_list[0][closing_subset_outer]
    # df_close_inner = allStock_df_list[0][closing_subset_inner]
    # df_close = allStock_df_list[0][closing_subset]

    # #open
    # opening_subset_outer = ['Apple','Amazon','Altaba','United Health','Home Depot']
    # opening_subset_inner = ['United Technologies','3M','JPMorgan Chase','Cisco Systems','American Express']
    # opening_subset = opening_subset_inner + opening_subset_outer

    # df_open_outer = allStock_df_list[0][opening_subset_outer]
    # df_open_inner = allStock_df_list[0][opening_subset_inner]
    # df_open = allStock_df_list[0][opening_subset]

    # #open-close
    # diff_subset_outer = ['Apple','Amazon','Altaba','United Health','Nike']
    # diff_subset_inner = ['United Technologies','3M','General Electric','IBM','American Express']
    # diff_subset = diff_subset_inner + diff_subset_outer

    # df_diff_outer = allStock_df_list[0][diff_subset_outer]
    # df_diff_inner = allStock_df_list[0][diff_subset_inner]
    # df_diff = allStock_df_list[0][diff_subset]

    # #high
    # high_subset_outer = ['Procter & Gamble','Walmart','Merk','United Health','Verizon']
    # high_subset_inner = ['JPMorgan Chase','3M','General Electric','IBM','American Express']
    # high_subset = high_subset_inner + high_subset_outer

    # df_high_outer = allStock_df_list[0][high_subset_outer]
    # df_high_inner = allStock_df_list[0][high_subset_inner]
    # df_high = allStock_df_list[0][high_subset]

    # #low
    # low_subset_outer = ['Apple','Amazon','Altaba','Verizon','Procter & Gamble']
    # low_subset_inner = ['United Technologies','3M','General Electric','Walt Disney','American Express']
    # low_subset = low_subset_inner + low_subset_outer

    # df_low_outer = allStock_df_list[0][low_subset_outer]
    # df_low_inner = allStock_df_list[0][low_subset_inner]
    # df_low = allStock_df_list[0][low_subset]


    # #combining stocks
    # allStock_df_list_inner = [df_close_inner, df_open_inner,df_diff_inner, df_high_inner, df_low_inner]
    # allStock_df_list_outer = [df_close_outer, df_open_outer,df_diff_outer, df_high_outer, df_low_outer]
    # allStock_df_list = [df_close, df_open,df_diff, df_high, df_low]

    allStock_df_list_inner, allStock_df_list_outer, allStock_df_list = subsetStocks(allStock_df_list)


    #inner
    corrNet_close, corrNet_open, corrNet_close_diff, corrNet_high, corrNet_low = correlation_networks(allStock_df_list_inner)

    # print(type(corrNet_close))
    # print('llllllllllllllllllll')

    plot_network_graphs(corrNet_close, corrNet_open, corrNet_close_diff, corrNet_high, corrNet_low,'inner')

    #outer
    corrNet_close, corrNet_open, corrNet_close_diff, corrNet_high, corrNet_low = correlation_networks(allStock_df_list_outer)

    plot_network_graphs(corrNet_close, corrNet_open, corrNet_close_diff, corrNet_high, corrNet_low,'outer')

    #combined
    corrNet_close, corrNet_open, corrNet_close_diff, corrNet_high, corrNet_low = correlation_networks(allStock_df_list)

    plot_network_graphs(corrNet_close, corrNet_open, corrNet_close_diff, corrNet_high, corrNet_low,'combined')


def run_risk_graphs(allStock_df_list):
    allStock_df_list_inner, allStock_df_list_outer, allStock_df_list = subsetStocks(allStock_df_list)

    corrNet_close, corrNet_open, corrNet_close_diff, corrNet_high, corrNet_low = correlation_networks(allStock_df_list)

    risk_graph(corrNet_high,'high_combined')

    





def networkGraphsMainFunction(allStock_df_list):
    full_stock_sets(allStock_df_list)
    print('Full set done')
    select_stock_set(allStock_df_list)
