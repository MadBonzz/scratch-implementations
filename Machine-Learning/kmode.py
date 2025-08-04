import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 10
np.random.seed(SEED)

df = pd.read_excel('Walmart_Store_sales.xlsx')
df.drop(columns=['Date', 'Store'], inplace=True)
k = 5

idxs = np.random.randint(0, df.shape[0], k)

samples = [df.iloc[idx].to_dict() for idx in idxs]

clusters = {k : v for k, v in enumerate(samples)}

def compare_samples(s1, s2, columns):
    difference = 0
    for col in columns:
        difference += (s1[col] - s2[col])
    return difference

def get_allocations(df, clusters):
    allocations = {}
    for i in range(len(clusters.keys())):
        allocations[i] = []
    for idx, row in df.iterrows():
        min_distance = float('inf')
        min_idx = -1
        for i in range(len(clusters.keys())):
            dist = compare_samples(row, clusters[i], df.columns)
            dist = abs(dist)
            if dist < min_distance:
                min_distance = dist
                min_idx = i
        allocations[min_idx].append(idx)
    return allocations

def get_new_clusters(clusters):
    new_clusters = {}
    cluster_idxs = get_allocations(df, clusters)
    for i in cluster_idxs.keys():
        idxs = cluster_idxs[i]
        temp_df = df.loc[idxs]
        modes = temp_df.mode().iloc[0]
        new_clusters[i] = modes.to_dict()
    return new_clusters, list(cluster_idxs.values())

n_iters = 5

print("Initial clusters were : ")
for i in clusters.keys():
    print(f"Cluster {i+1} is : {clusters[i]}")

curr_allocs = []

while(True):
    clusters, new_allocs = get_new_clusters(clusters)
    if curr_allocs == new_allocs:
        break
    else:
        new_allocs = curr_allocs

print("Final clusters are : ")
for i in clusters.keys():
    print(f"Cluster {i+1} is : {clusters[i]}")
