import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(path="data/raw/supermarket_sales.csv"):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    df = df.copy()
    df = df.drop_duplicates()
    if df.isnull().sum().sum() > 0:
        df = df.fillna(method='ffill')
    return df

def engineer_features(df):
    df = df.copy()
    df['avg_price_per_item'] = df['Total'] / (df['Quantity'] + 1e-6)
    df['profit_margin'] = df['gross income'] / (df['Total'] + 1e-6)
    df['day_of_week'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['hour'] = pd.to_datetime(df['Date'] + ' ' + df['Time']).dt.hour
    return df

def get_preprocessor(numeric_features, categorical_features):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

def prepare_data(df, numeric_features, categorical_features):
    preprocessor = get_preprocessor(numeric_features, categorical_features)
    
    X = preprocessor.fit_transform(df[numeric_features + categorical_features])
    
    feature_names = (
        list(numeric_features) +
        list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    )
    
    return X, feature_names, preprocessor

def run_preprocessing(path="data/raw/supermarket_sales.csv"):
    df = load_data(path)
    df = clean_data(df)
    df = engineer_features(df)
    
    numeric_features = ['Unit price', 'Quantity', 'Total', 'Tax 5%', 
                        'cogs', 'gross margin percentage', 'gross income', 
                        'avg_price_per_item', 'profit_margin', 'day_of_week', 'hour']
    
    categorical_features = ['Branch', 'City', 'Customer type', 'Gender', 
                           'Product line', 'Payment']
    
    X, feature_names, preprocessor = prepare_data(df, numeric_features, categorical_features)
    
    return X, feature_names, df, preprocessor

if __name__ == "__main__":
    X, feature_names, df, preprocessor = run_preprocessing()
    print(f"Processed shape: {X.shape}")
    print(f"Feature names: {len(feature_names)} features")
    print(f"Sample transformed data shape: {X[:3].shape}")