import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

# calculates the communicability betweeness centrality and returns a dictionary
risk_alloc = nx.communicability_betweenness_centrality(H_master)

# converts the dictionary of degree centralities to a pandas series
risk_alloc = pd.Series(risk_alloc)

# normalizes the degree centrality 
risk_alloc = risk_alloc / risk_alloc.sum()

# resets the index
risk_alloc.reset_index(inplace=True)

# converts series to a sorted DataFrame
risk_alloc = (
    pd.DataFrame({"Stocks": risk_alloc['index'], "Risk Allocation": risk_alloc.values})
        .sort_values(by="Risk Allocation", ascending=True)
        .reset_index()
        .drop("index", axis=1)
)

# initializes figure
fig, ax = plt.subplots(figsize=(8,10))

# creates a bar plot
ax.barh(y=risk_alloc['Stocks'], width=risk_alloc['Risk Allocation'], color='purple')

# sets the x axis label
ax.set_xlabel('Relative Risk %', fontsize=12)

# sets the y axis label
ax.set_ylabel('Historical Portfolio (2006-2014)', fontsize=12)

# sets the title
ax.set_title('Intraportfolio Risk', fontsize=18)

# iterates over the stocks (label) and their numerical index (i)
for i, label in enumerate(list(risk_alloc.index)):
    # gets the relative risk as a percentage (the labels)
    value = (risk_alloc.loc[label, 'Risk Allocation']*100).round(2).astype(str) + '%'
    # annotates the bar plot with the relative risk percentages
    ax.text(x=risk_alloc.loc[label, 'Risk Allocation']+0.001, y=i+0.15, s=value)

# removes spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# turns xticks off
ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

# show the plot
plt.show()
