import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from preprocessing import load_data, clean_data, engineer_features, run_preprocessing

print("=" * 60)
print("COMPARING K=2 vs K=3 CLUSTERING")
print("=" * 60)

X, feature_names, df, preprocessor = run_preprocessing()

# --- K=2 ---
print("\n--- K=2 CLUSTERS ---")
kmeans2 = KMeans(n_clusters=2, random_state=42, n_init=10)
labels2 = kmeans2.fit_predict(X)

df_eval2 = df.copy()
df_eval2['cluster'] = labels2
crosstab2 = pd.crosstab(df_eval2['cluster'], df_eval2['Customer type'])
print("Cluster vs Customer type:")
print(crosstab2)

purity2 = 0
n = len(df_eval2)
for i in range(2):
    cluster_i = df_eval2[df_eval2['cluster'] == i]
    majority_count = cluster_i['Customer type'].value_counts().iloc[0]
    purity2 += majority_count / n
purity2 = purity2 / 2
print(f"Purity: {purity2:.4f}")
print(f"Silhouette Score: {silhouette_score(X, labels2):.4f}")

cluster_sizes2 = pd.Series(labels2).value_counts().sort_index().to_dict()
print(f"Cluster sizes: {cluster_sizes2}")

# Average totals
df_orig2 = pd.read_csv(r"data\raw\supermarket_sales.csv")
df_orig2["cluster"] = labels2
for i in range(2):
    avg_total = df_orig2[df_orig2["cluster"] == i]["Total"].mean()
    print(f"Cluster {i} avg Total: ${avg_total:.2f}")

# --- K=3 ---
print("\n--- K=3 CLUSTERS ---")
kmeans3 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels3 = kmeans3.fit_predict(X)

df_eval3 = df.copy()
df_eval3["cluster_3"] = labels3
crosstab3 = pd.crosstab(df_eval3["cluster_3"], df_eval3["Customer type"])
print("Cluster vs Customer type:")
print(crosstab3)

purity3 = 0
n = len(df_eval3)
for i in range(3):
    cluster_i = df_eval3[df_eval3["cluster_3"] == i]
    majority_count = cluster_i["Customer type"].value_counts().iloc[0]
    purity3 += majority_count / n
purity3 = purity3 / 3
print(f"Purity: {purity3:.4f}")
print(f"Silhouette Score: {silhouette_score(X, labels3):.4f}")

cluster_sizes3 = pd.Series(labels3).value_counts().sort_index().to_dict()
print(f"Cluster sizes: {cluster_sizes3}")

# Average totals
df_orig3 = pd.read_csv(r"data\raw\supermarket_sales.csv")
df_orig3["cluster_3"] = labels3
for i in range(3):
    avg_total = df_orig3[df_orig3["cluster_3"] == i]["Total"].mean()
    print(f"Cluster {i} avg Total: ${avg_total:.2f}")

# --- SUMMARY COMPARISON ---
print("\n" + "=" * 60)
print("SUMMARY COMPARISON")
print("=" * 60)
print(f"{'Metric':<20} {'k=2':<15} {'k=3':<15}")
print("-" * 50)
print(f"{'Purity vs Customer':<20} {purity2:.4f}         {purity3:.4f}")
print(f"{'Silhouette Score':<20} {silhouette_score(X, labels2):.4f}     {silhouette_score(X, labels3):.4f}")
best_k2 = max(cluster_sizes2, key=cluster_sizes2.get)
worst_k2 = min(cluster_sizes2, key=cluster_sizes2.get)
best_k3 = max(cluster_sizes3, key=cluster_sizes3.get)
worst_k3 = min(cluster_sizes3, key=cluster_sizes3.get)
print(f"{'Best cluster size':<20} {best_k2}                  {best_k3}")
print(f"{'Worst cluster size':<20} {worst_k2}                {worst_k3}")