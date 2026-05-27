# dataset.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def data_splitting (data_path="reviews_clean.jsonl", test_size=0.2, random_state=42):

    # Validate the data file path and ensure it exists
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path.resolve()}")
        
    # Load the dataset and ensure the correct columns are present
    df = pd.read_json(data_path, lines=True)
    if 'content' not in df.columns or 'rating_label' not in df.columns:
        raise ValueError("Dataset must contain 'content' and 'rating_label' columns")
    
    # Extract features and labels
    X = df['content'].astype(str)
    y = df['rating_label']
    
    # Perform stratified splitting to maintain class distribution
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    return X_train_raw, X_test_raw, y_train, y_test