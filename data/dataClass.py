from enum import Enum
import matplotlib.pyplot as plt
import pandas as pd


class DataType(Enum):
    """# DataType
    The data types for the different kinds of base stock data

    ## Values:
    - DOW (str): Dow Jones adjusted closing prices
    - ETFS (str): ETFs
    - IBB (str): IBB Index
    - NG (str): Network Graph Data
    - SPX (str): SPX Index
    """

    DOW = "Dow_Adj"
    ETFS = "ETFs"
    IBB = "IBB_Index"
    NG = "Network_Graph_Data"
    SPX = "SPX_Index"

    def __str__(self) -> str:
        return self.value


class Data:
    """# Data
    The data and properties class for stock data

    ## Arguments
    - df (pd.DataFrame): The data frame of the data
    - dType (DataType): The data type of the dataset
    - name (str): The name of the dataset
    - perc (float): Percent value between 0 and 1 to split for validation.
        Defaults to 20%

    ## Properties
    - df (pd.DataFrame): The data frame of the data
    - dType (DataType): The data type of the dataset
    - name (str): The name of the dataset
    - perc (float): Percent value between 0 and 1 to split for validation
    - train_df (pd.DataFrame): The data frame for the training data
    - validate_df (pd.DataFrame): The data frame for the validation data

    ## Methods
    - `plot()`: Plot the full data frame using pandas built in plotting
    """

    def __init__(self, df: pd.DataFrame, dType: DataType, name: str, perc: float = 0.2):
        self._df: pd.DataFrame = df
        self._dType: DataType = dType
        self._name: str = name
        self._perc: float = perc
        self._train_df: pd.DataFrame = None
        self._validate_df: pd.DataFrame = None

        #self._clean_data()
        self._split_data(perc=self.perc)

    def get_plot(self) -> plt.Figure:
        """Return a configured `plt.Figure` to graph"""
        
        fig = plt.figure()
        new_plot = fig.add_subplot(111)
        new_plot.plot(self.df)

        plt.close(fig)

        return fig

    def _clean_data(self) -> pd.DataFrame:
        """Drop any rows of the full data frame that contain NA"""

        self._df = self.df.dropna(axis=1, how='any')
        return self.df

    def _split_data(self, perc: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """# split_data
        Splits the full dataframe into training and validation
        datasets by slicing the end of the series.

        ## Arguments
        - perc (float): Percent value between 0 and 1 to split for validation

        ## Returns
            The training and validation dataframes, respectfully.
        """

        if self.train_df is None or self.validate_df is None:
            ind = int(len(self.df.index)*perc)
            
            self._train_df = self.df.iloc[:-ind, :]
            self._validate_df = self.df.iloc[-ind:, :]

        return self.train_df, self.validate_df

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def dType(self) -> DataType:
        return self._dType

    @property
    def name(self) -> str:
        return self._name

    @property
    def perc(self) -> float:
        return self._perc

    @property
    def train_df(self) -> pd.DataFrame:
        return self._train_df

    @property
    def validate_df(self) -> pd.DataFrame:
        return self._validate_df


class IBBData(Data):
    """# IBBData
    Extends `Data` to add three `Data` objects for last price, net-change, and percent change.

    ## Arguments
    - df (pd.DataFrame): The data frame of the IBB Index
    - df_lp (pd.DataFrame): The data frame of the IBB Stock's last price
    - df_nc (pd.DataFrame): The data frame of the IBB Stock's net-change
    - df_pc (pd.DataFrame): The data frame of the IBB Stock's percent change
    - name (str): The name of the dataset
    - perc (float): Percent value between 0 and 1 to split for validation.
        Defaults to 20%

    ## Properties
    - Inherits all the properties from `Data`
    - lp_data (dataClass.Data): The `Data` class for the IBB Stocks' last price
    - nc_data (dataClass.Data): The `Data` class for the IBB Stocks' net-change
    - pc_data (dataClass.Data): The `Data` class for the IBB Stocks' percent change
    """

    def __init__(self, df: pd.DataFrame, df_lp: pd.DataFrame, df_nc: pd.DataFrame, df_pc: pd.DataFrame,
                 name: str, perc: float = 0.2):
        super().__init__(df, DataType.IBB, name, perc)
        self._lp_data= Data(df_lp, DataType.IBB, "IBB Stocks' Last Price Data")
        self._nc_data = Data(df_nc, DataType.IBB, "IBB Stocks' Net Change Data")
        self._pc_data = Data(df_pc, DataType.IBB, "IBB Stocks' Percent Change Data")

    def plot(self):
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
        axs[0,0].set_figure(self.df.plot().get_figure())
        axs[0,1].set_figure(self.lp_data.df.plot().get_figure())
        axs[1,0].set_figure(self.nc_data.df.plot().get_figure())
        axs[1,1].set_figure(self.pc_data.df.plot().get_figure())
        
        plt.show()

    @property
    def lp_data(self) -> Data:
        return self._lp_data

    @property
    def nc_data(self) -> Data:
        return self._nc_data

    @property
    def pc_data(self) -> Data:
        return self._pc_data
