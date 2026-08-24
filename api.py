import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from preprocessing import run_preprocessing

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({
        'status': 'ok',
        'message': 'ML API is running',
        'endpoints': [
            '/api/cluster?k=2',
            '/api/evaluate?k=2',
            '/cluster?k=2',
            '/evaluate?k=2'
        ]
    })

@app.route('/api/cluster')
def api_cluster():
    k = int(request.args.get('k', 2))
    
    try:
        X, feature_names, df, preprocessor = run_preprocessing()
        
        # Train K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Calculate silhouette score
        sil_score = silhouette_score(X, labels)
        
        # Get cluster stats
        df['cluster'] = labels
        cluster_stats = df.groupby('cluster').mean(numeric_only=True)
        
        # Prepare results
        results = {
            'k': k,
            'silhouette_score': sil_score,
            'n_samples': len(df),
            'cluster_distribution': df['cluster'].value_counts().to_dict(),
            'profiles': {}
        }
        
        # Add cluster profiles
        for i in range(k):
            if i in cluster_stats.index:
                results['profiles'][f'cluster_{i}'] = {
                    'avg_total': float(cluster_stats.loc[i, 'Total']),
                    'avg_quantity': float(cluster_stats.loc[i, 'Quantity']),
                    'avg_unit_price': float(cluster_stats.loc[i, 'Unit price']),
                    'avg_rating': float(cluster_stats.loc[i, 'Rating']),
                }
            else:
                results['profiles'][f'cluster_{i}'] = {
                    'avg_total': 0,
                    'avg_quantity': 0,
                    'avg_unit_price': 0,
                    'avg_rating': 0,
                }
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/evaluate')
def api_evaluate():
    k = int(request.args.get('k', 2))
    
    try:
        X, feature_names, df, preprocessor = run_preprocessing()
        
        # Train K-Means with different k values
        scores = {}
        K_range = range(2, min(k + 3, 9))
        
        for k_test in K_range:
            kmeans_test = KMeans(n_clusters=k_test, random_state=42, n_init=10)
            labels_test = kmeans_test.fit_predict(X)
            scores[k_test] = silhouette_score(X, labels_test)
        
        # Find best k
        best_k = max(scores, key=scores.get)
        best_score = scores[best_k]
        
        # Get cluster vs Customer type crosstab
        kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels_final = kmeans_final.fit_predict(X)
        df['cluster'] = labels_final
        crosstab = pd.crosstab(df['cluster'], df.get('Customer type', 'N/A')).to_dict()
        
        return jsonify({
            'success': True,
            'elbow_data': scores,
            'best_k': best_k,
            'best_silhouette': best_score,
            'cluster_vs_customer': crosstab
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/cluster')
def cluster():
    '''Legacy endpoint - same as /api/cluster'''
    return api_cluster()

@app.route('/evaluate')
def evaluate():
    '''Legacy endpoint - same as /api/evaluate'''
    return api_evaluate()