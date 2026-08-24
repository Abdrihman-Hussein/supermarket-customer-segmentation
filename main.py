import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from preprocessing import load_data, clean_data, engineer_features, run_preprocessing

def find_optimal_k(X, k_range=(2, 10)):
    """Find optimal k using silhouette score"""
    scores = []
    K_range = range(k_range[0], k_range[1] + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        scores.append(score)
        print(f"k={k}: Silhouette Score = {score:.4f}")
    
    best_k = K_range[np.argmax(scores)]
    return K_range, scores, best_k

def train_final_model(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return kmeans, labels

def analyze_clusters(df, labels, feature_names):
    df = df.copy()
    df['cluster'] = labels
    
    cluster_means = df.groupby('cluster').mean(numeric_only=True)
    return cluster_means

def evaluate_clusters_against_ground_truth(df, labels, ground_truth_col='Customer type'):
    """Evaluate how well clusters align with known Customer type labels"""
    evaluation = df.copy()
    evaluation['cluster'] = labels
    evaluation[ground_truth_col] = df[ground_truth_col]
    
    # Cross-tabulation: cluster vs Customer type
    crosstab = pd.crosstab(evaluation['cluster'], evaluation[ground_truth_col])
    print("\nCluster vs Customer type cross-tabulation:")
    print(crosstab)
    
    # Calculate purity: for each cluster, what's the majority class?
    purity = 0
    for cluster_id in range(labels.max() + 1):
        cluster_labels = evaluation[evaluation['cluster'] == cluster_id][ground_truth_col]
        majority_count = cluster_labels.value_counts().iloc[0]
        purity += majority_count / len(evaluation)
    
    purity = purity / (labels.max() + 1)
    print(f"\nCluster purity against Customer type: {purity:.4f}")
    return crosstab, purity

def train_and_evaluate_pipeline():
    print("=" * 50)
    print("STEP 1: Loading and preprocessing data...")
    X, feature_names, df, preprocessor = run_preprocessing()
    print(f"Data prepared: {X.shape} samples, {X.shape[1]} features (already encoded+scaled)")
    
    print("\n" + "=" * 50)
    print("STEP 2: Splitting preprocessed data into train/test...")
    # Split the already-preprocessed feature matrix X
    X_train, X_test, y_train, y_test = train_test_split(X, df['Customer type'], test_size=0.3, random_state=42, shuffle=True)
    print(f"Train set: {X_train.shape[0]} transactions")
    print(f"Test set: {X_test.shape[0]} transactions")
    
    print("\n" + "=" * 50)
    print("STEP 3: Finding optimal k on TRAIN data...")
    k_range, scores, best_k = find_optimal_k(X_train, k_range=(2, 8))
    
    print(f"\nBest k on train data: {best_k} with silhouette score: {max(scores):.4f}")
    
    print("\n" + "=" * 50)
    print("STEP 4: Training final model on TRAIN data...")
    kmeans, labels_train = train_final_model(X_train, best_k)
    
    print("\n" + "=" * 50)
    print("STEP 5: Predicting clusters on TEST data...")
    test_cluster_labels = kmeans.predict(X_test)
    print(f"Test set cluster assignments: {len(test_cluster_labels)} transactions")
    
    print("\n" + "=" * 50)
    print("STEP 6: Evaluating clusters on TEST data vs ground truth...")
    # Create evaluation dataframe with customer type and cluster labels
    test_df_eval = pd.DataFrame({
        'Customer type': y_test.values,
        'cluster': test_cluster_labels
    })
    
    crosstab, purity = evaluate_clusters_against_ground_truth(test_df_eval, test_cluster_labels, 'Customer type')
    
    print("\n" + "=" * 50)
    print("STEP 7: Analyzing full dataset clusters...")
    kmeans_full, labels_full = train_final_model(X, best_k)
    cluster_means = analyze_clusters(df, labels_full, feature_names)
    print("Cluster Profiles (on all data):")
    print(cluster_means)
    
    print("\n" + "=" * 50)
    print("STEP 8: Saving results...")
    import os
    os.makedirs("results", exist_ok=True)
    full_df_with_clusters = df.copy()
    full_df_with_clusters['cluster'] = labels_full
    full_df_with_clusters.to_csv("results/clusters_train_test_split.csv", index=False)
    print("Results saved to results/clusters_train_test_split.csv")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_range, scores, 'bo-')
    plt.axvline(best_k, color='red', linestyle='--')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Elbow Method on Train Data')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Cluster vs Customer Type\nPurity: {purity:.2f}')
    plt.ylabel('Cluster')
    plt.xlabel('Customer Type')
    
    plt.tight_layout()
    plt.savefig('results/train_test_evaluation.png')
    plt.close()
    print("Evaluation plot saved to results/train_test_evaluation.png")
    
    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE!")
    print(f"\nKey Findings:")
    print(f"- Best k: {best_k}")
    print(f"- Cluster purity against Customer type: {purity:.4f}")
    print(f"- This shows how well the clustering 'understands' the customer types")
    print(f"- Higher purity means clusters better capture Member/Normal patterns")

if __name__ == "__main__":
    train_and_evaluate_pipeline()

if __name__ == "__main__":
    train_and_evaluate_pipeline()