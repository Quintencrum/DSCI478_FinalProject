import pandas as pd

def clean_data(df: pd.dataframe) -> pd.dataframe:
    '''This method drops any rows of data that contains NA.

    # Arguments
     - df (pd.dataframe): The dataframe to clean

    # Returns
        This method returns a cleaned dataframe.
    '''
    df = df.dropna(axis=1, how='any')
    return df

def split_data(df: pd.dataframe, perc: float) -> tuple[pd.dataframe,pd.dataframe]:
    '''This method splits the dataframe into training and validation
    datasets by slicing the end of the series.

    # Arguments
     - df (pd.dataframe): The dataframe to split
     - perc (float): Percent value between 0 and 1 to split for validation

    # Returns
        This method returns a training and validation dataframe respectfully.
    '''
    ind = int(len(df.index)*perc)
    
    train_df = df.iloc[:-ind, :]
    vald_df = df.iloc[-ind:, :]

    return train_df, vald_df
