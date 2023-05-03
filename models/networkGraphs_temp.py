import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

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
