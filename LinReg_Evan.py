# Code for Final Models
import numpy as np
import pandas as pd
import matplotlib as plt

from sklearn.linear_model import LinearRegression, Lasso, Ridge

# Loading data frames from Github -- df = Our df
df = pd.read_csv('https://raw.githubusercontent.com/Quintencrum/DSCI478_FinalProject/Q/data/Network_Graphs/all_stocks.csv')
returns = pd.read_csv('https://raw.githubusercontent.com/firmai/techniques/master/Assets/returns.csv')
ibb = pd.read_csv('https://raw.githubusercontent.com/Quintencrum/DSCI478_FinalProject/main/data/Deep_Portfolio/ibb.csv')

# dataframe manipulation
df['return'] = df['Open'] - df['Close']
df1 = df.drop(['Open', 'High', 'Low', 'Close','Volume'], axis=1)

df1['Name'].unique()
df2 = df1.pivot(index='Date', columns='Name', values='return')
df2 = df2.fillna(0)
df2['label'] = 1

df2 = df2.reset_index()
df2 = df2.drop(['Date'], axis=1) # dates starting in 2006-01-03, going till 2017-12-29

## Beginning of Regression Section
# Normal Linear Regression Weights
reg = LinearRegression(fit_intercept=False).fit(df2.drop(["label"],axis=1), df2["label"])

coeffs = pd.DataFrame(df2.drop(['label'], axis=1).columns)
coeffs.columns = ["Tickers"]
coeffs["Coefficients"] = reg.coef_
coeffs["Weights"] = (coeffs["Coefficients"].abs() / coeffs["Coefficients"].abs().sum())*100
coeffs.sort_values('Weights', ascending=False).head(7)


# Lasso Regression Weights

reg = Lasso(fit_intercept=False,alpha=0.00005, positive=True).fit(df2.drop(["label"],axis=1), df2["label"])

lasso_coeffs = pd.DataFrame(df2.drop(["label"],axis=1).columns)
lasso_coeffs.columns = ["Tickers"]
lasso_coeffs["Coefficients"] = reg.coef_
lasso_coeffs["Weights"] = (lasso_coeffs["Coefficients"].abs() / lasso_coeffs["Coefficients"].abs().sum())*100

lasso_coeffs.sort_values('Weights', ascending=False).head(7) # reduces ALL others to zero

# Ridge Regression Weights

reg = Ridge(fit_intercept=False,alpha=1, positive=True).fit(df2.drop(["label"],axis=1), df2["label"])

ridge_coeffs = pd.DataFrame(df2.drop(["label"],axis=1).columns)
ridge_coeffs.columns = ["Tickers"]
ridge_coeffs["Coefficients"] = reg.coef_
ridge_coeffs["Weights"] = (ridge_coeffs["Coefficients"].abs() / ridge_coeffs["Coefficients"].abs().sum())*100

ridge_coeffs.sort_values('Weights', ascending=False).head(7) # reduces ALL others to zero

## End of Regression for Our Dataframe

## Beginning Regression on Their Dataframe ('return' dataframe)

returns = returns.set_index('date', drop=True) # set index correctly

returns["label"] = 1 

# Normal Regression Section
reg = LinearRegression(fit_intercept=False).fit(returns.drop(["label"],axis=1), returns["label"])

normal_coeffs = pd.DataFrame(returns.drop(["label"],axis=1).columns)
normal_coeffs.columns = ["TICKER"]
normal_coeffs["coeff"] = reg.coef_
normal_coeffs["weight"] = (normal_coeffs["coeff"].abs() / normal_coeffs["coeff"].abs().sum())*100

print("")
print("Normal Regression Weights")
print(normal_coeffs.sort_values('weight', ascending=False).head(5))

# Lasso Regression Section
reg = Lasso(fit_intercept=False,alpha=0.0005, positive=True).fit(returns.drop(["label"],axis=1), returns["label"])

lasso_coeffs = pd.DataFrame(returns.drop(["label"],axis=1).columns)
lasso_coeffs.columns = ["TICKER"]
lasso_coeffs["coeff"] = reg.coef_
lasso_coeffs["weight"] = (lasso_coeffs["coeff"].abs() / lasso_coeffs["coeff"].abs().sum())*100


print("Lasso Regression Weights, alpha=0.0005")
print(lasso_coeffs.sort_values('weight', ascending=False).head(5))

# Ridge Regression Section
reg = Ridge(fit_intercept=False,alpha=1).fit(returns.drop(["label"],axis=1), returns["label"])

ridge_coeffs = pd.DataFrame(returns.drop(["label"],axis=1).columns)
ridge_coeffs.columns = ["TICKER"]
ridge_coeffs["coeff"] = reg.coef_
ridge_coeffs["weight"] = (ridge_coeffs["coeff"].abs() / ridge_coeffs["coeff"].abs().sum())*100

print("")
print("Ridge Regression Weights")
print(ridge_coeffs.sort_values('weight', ascending=False).head(5))


## End of Regression for Their Dataframes

## Data Manipulation for Graphing

# Our dataframes
normal_df = df2
lasso_df = df2
ridge_df = df2

# Their dataframes
normal_returns = returns
lasso_returns = returns
ridge_returns = returns

#Our Datasets top 5 weighted stocks
normal_df = normal_df[['GE','VZ','MSFT','INTC','MMM']]
lasso_df = lasso_df[['GE','AABA','GOOGL','KO','WMT']]
ridge_df = ridge_df[['GOOGL','AABA','GE','BA','CAT']]

# Their datasets top 5 weighted stocks
normal_returns = normal_returns[['NURO','NFLX','SCON','RBCN','HHS']]
lasso_returns = lasso_returns[['RGCO','FCAP','DLTR','HIFS','EXPO']]
ridge_returns = ridge_returns[['NURO','NFLX','SCON','RBCN','HHS']]

# Data point reduction, 15 data points total (helps make graphs more readable)
normal_df = normal_df.loc[::201] # 15 data points
lasso_df = lasso_df.loc[::201] # 15 data points
ridge_df = ridge_df.loc[::201] # 15 data points

normal_returns = normal_returns.loc[::9] # 15 data pts - sampled through at random uniform steps
lasso_returns = lasso_returns.loc[::9] # 15 data pts
ridge_returns = ridge_returns.loc[::9] # 15 data pts

# for comparison to ibb index
ibb = ibb.loc[::15] # 15 data pts

## Beginning of Calculating and Plotting Returns

# Calculations --> Normal Linear Regression
GE = normal_df['GE'].sum().round(4)
VZ = normal_df['VZ'].sum().round(4)
MSFT = normal_df['MSFT'].sum().round(4)
INTC = normal_df['INTC'].sum().round(4)
MMM = normal_df['MMM'].sum().round(4)

# sum of total normal returns - us
normal_df_returns = (GE+VZ+MSFT+INTC+MMM).round(4)

print('Total Returns GE:   ', GE)
print('Total Returns VZ:   ', VZ)
print('Total Returns MSFT: ', MSFT)
print('Total Returns INTC: ', INTC)
print('Total Returns MMM:  ', MMM)
print('')
print('Total Sum Returns:  ', normal_df_returns)

# Plot of Returns for Normal DF
normal_df.plot(use_index=True, y=['GE', 'VZ', 'MSFT', 'INTC', 'MMM'], kind='line')


#  Calculations --> LASSO Regression
GE = lasso_df['GE'].sum().round(4)
AABA = lasso_df['AABA'].sum().round(4)
GOOGL = lasso_df['GOOGL'].sum().round(4)
KO = lasso_df['KO'].sum().round(4)
WMT = lasso_df['WMT'].sum().round(4)

# sum of total lasso returns
lasso_df_returns = (GE+AABA+GOOGL+KO+WMT).round(4)

print('Total Returns GE:   ', GE)
print('Total Returns AABA: ', AABA)
print('Total Returns GOOGL:', GOOGL)
print('Total Returns KO:   ', KO)
print('Total Returns WMT:  ', WMT)
print('')
print('Total Sum Returns:  ', lasso_df_returns)

# Plot of Returns for Lasso DF
lasso_df.plot(use_index=True, y=['GE', 'AABA', 'GOOGL', 'KO', 'WMT'], kind='line')


# Calculations and Returns --> Ridge Regression
GE = ridge_df['GE'].sum().round(4)
AABA = ridge_df['AABA'].sum().round(4)
GOOGL = ridge_df['GOOGL'].sum().round(4)
BA = ridge_df['BA'].sum().round(4)
CAT = ridge_df['CAT'].sum().round(4)

# sum of ridge returns
ridge_df_returns = (GE+AABA+GOOGL+BA+CAT).round(4)

print('Total Returns GOOGL:', GOOGL)
print('Total Returns AABA: ', AABA)
print('Total Returns GE:   ', GE)
print('Total Returns BA:   ', BA)
print('Total Returns CAT:  ', CAT)
print('')
print('Total Sum Returns:  ', ridge_df_returns)

# Plot of Returns for Ridge DF
ridge_df.plot(use_index=True, y=['GOOGL', 'AABA', 'GE', 'BA', 'CAT'], kind='line')



## Regression Calculations --> Normal Returns
NURO = normal_returns['NURO'].sum().round(4)
NFLX = normal_returns['NFLX'].sum().round(4)
SCON = normal_returns['SCON'].sum().round(4)
RBCN = normal_returns['RBCN'].sum().round(4)
HHS = normal_returns['HHS'].sum().round(4)

# sum of their normal returns
normal_returns_test = (NURO+NFLX+SCON+RBCN+HHS).round(4)

# Printing Returns
print('Total Returns NURO:', NURO)
print('Total Returns NFLX:', NFLX)
print('Total Returns SCON:', SCON)
print('Total Returns RBCN:', RBCN)
print('Total Returns HHS: ', HHS)
print("")
print('Total Returns Test:', normal_returns_test)

# Plot of Returns for Normal Returns
normal_returns.plot(use_index=True, y=['NURO', 'NFLX', 'SCON', 'RBCN', 'HHS'], kind='line')


## Returns based on their datasets --> LASSO Regression
RGCO = lasso_returns['RGCO'].sum().round(4)
FCAP = lasso_returns['FCAP'].sum().round(4)
DLTR = lasso_returns['DLTR'].sum().round(4)
HIFS = lasso_returns['HIFS'].sum().round(4)
EXPO = lasso_returns['EXPO'].sum().round(4)

# sum of their lasso returns
lasso_returns_test = (RGCO+FCAP+DLTR+HIFS+EXPO).round()

print('Total Returns RGCO: ', RGCO)
print('Total Returns FCAP: ', FCAP)
print('Total Returns DLTR: ', DLTR)
print('Total Returns HIFS: ', HIFS)
print('Total Returns EXPO: ', EXPO)
print('')
print('Total Lasso Returns:', lasso_returns_test)

# Plot of Returns for Lasso Returns
lasso_returns.plot(use_index=True, y=['RGCO', 'FCAP', 'DLTR', 'HIFS', 'EXPO'], kind='line')


## Return Calculations from their DF --> Ridge Regression
NURO = ridge_returns['NURO'].sum().round(4)
NFLX = ridge_returns['NFLX'].sum().round(4)
SCON = ridge_returns['SCON'].sum().round(4)
RBCN = ridge_returns['RBCN'].sum().round(4)
HHS = ridge_returns['HHS'].sum().round(4)

# sum of their ridge returns
ridge_returns_test = (NURO+NFLX+SCON+RBCN+HHS).round(4)

# Printing Returns
print('Total Returns NURO:', NURO)
print('Total Returns NFLX:', NFLX)
print('Total Returns SCON:', SCON)
print('Total Returns RBCN:', RBCN)
print('Total Returns HHS: ', HHS)
print("")
print('Total Returns Test:', ridge_returns_test)

# Plot of Returns for Ridge Returns
ridge_returns.plot(use_index=True, y=['NURO', 'NFLX', 'SCON', 'RBCN', 'HHS'], kind='line')

## Comparison of Plots to IBB index
# Comparison to IBB Index
ibb = ibb.drop(['PX_LAST','% Change'], axis=1)

# Plot showing the ibb indexes Change
ibb.plot(use_index=True, y=['Change'], kind='line')
print('Total IBB Index Returns: ', ibb['Change'].sum().round(4))


# Final Notes -- 
# IBB Returns: 7.7
# Best Returns for Our DF: 20.97 (Ridge Reg)
# Best Returns for Their DF: 2.0 (LASSO Reg)
