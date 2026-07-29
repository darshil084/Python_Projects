import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go

data = pd.read_csv("CC_GENERAL.csv")
#print(data.head())
#print(data.isnull().sum())

data = data.dropna()

clustering_data = data[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]

scaler = MinMaxScaler()
clustering_data = scaler.fit_transform(clustering_data)

k_means = KMeans(n_clusters=5)
clusters = k_means.fit_predict(clustering_data)
data["CREDIT_CARD_SEGMENTS"] = clusters

data["CREDIT_CARD_SEGMENTS"] = data["CREDIT_CARD_SEGMENTS"].map({0:"Cluster 1",
                              1:"Cluster 2", 2:"Cluster 3", 3:"Cluster 4", 4:"Cluster 5"})
#print(data["CREDIT_CARD_SEGMENTS"].head(10))

PLOT = go.Figure()

for i in data["CREDIT_CARD_SEGMENTS"].unique():


    PLOT.add_trace(go.Scatter3d(
    x = data[data["CREDIT_CARD_SEGMENTS"] == i]["BALANCE"],
    y = data[data["CREDIT_CARD_SEGMENTS"] == i]["PURCHASES"],
    z = data[data["CREDIT_CARD_SEGMENTS"] == i]["CREDIT_LIMIT"],
    mode="markers",
    marker=dict(size=6, line=dict(width=1)),
    name=str(i)
    ))
    
PLOT.update_traces(hovertemplate= 'BALANCE: %{x} <br>PURCHASES: %{y} <br>CREDIT_LIMIT: %{z}')

PLOT.update_layout(width=800, height=600, autosize= True, showlegend= True,
                   scene = dict(
                                xaxis=dict(title=dict(text="BALANCE", font=dict(color="black"))),
                                yaxis=dict(title=dict(text="PURCHASES", font=dict(color="black"))),
                                zaxis=dict(title=dict(text="CREDIT_LIMIT", font=dict(color="black")))
                                ),              
                    font = dict(family= "Gilroy", size= 12, color= "Black"))

PLOT.show()
                                  
        