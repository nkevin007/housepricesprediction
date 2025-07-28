import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    df = df.dropna()
    if 'Neighborhood' in df.columns:
        le = LabelEncoder()
        df['Neighborhood'] = le.fit_transform(df['Neighborhood'])
    return df
