import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from datasets import Dataset



def prepare_finbert_data():
    """
    Prepare and preprocess data for FinBERT fine-tuning
    Returns: train_dataset, val_dataset, label_mapping
    """
    # Check if file exists
    csv_path = "data/all-data.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    df = None
    
    for encoding in encodings:
        try:
            # Read without headers since your CSV doesn't seem to have column names
            df = pd.read_csv(csv_path, encoding=encoding, header=None)
            print(f"Successfully read file with {encoding} encoding")
            print(f"Shape: {df.shape}")
            print("First few rows:")
            print(df.head())
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with {encoding}: {e}")
            continue
    
    if df is None:
        raise ValueError("Could not read CSV file with any encoding")
    
    # Check if we have at least 2 columns
    if df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns, but got {df.shape[1]}")
    
    # Assign column names based on your data format
    df.columns = ['sentiment', 'text'] + [f'extra_{i}' for i in range(2, df.shape[1])]
    
    # Keep only the first two columns (sentiment and text)
    df = df[['sentiment', 'text']]
    
    # Clean the data - remove any quotes or extra characters
    df['sentiment'] = df['sentiment'].str.replace('"', '').str.strip()
    df['text'] = df['text'].str.replace('"', '').str.strip()
    
    print(f"Unique sentiment values: {df['sentiment'].unique()}")
    print(f"Sample data:")
    print(df.head())
    
    # Prepare labels
    label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
    
    # Map sentiment to labels
    df['label'] = df['sentiment'].map(label_mapping)
    
    # Check for any unmapped values
    unmapped = df[df['label'].isna()]
    if not unmapped.empty:
        print(f"Warning: {len(unmapped)} rows with unmapped sentiment values:")
        print(unmapped['sentiment'].unique())
        df = df.dropna(subset=['label'])
    
    # Convert label to integer
    df['label'] = df['label'].astype(int)
    
    # Remove rows with missing values
    df = df.dropna(subset=['text', 'label'])
    
    print(f"Final dataset size: {len(df)}")
    print(f"Label distribution:")
    print(df['label'].value_counts())
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']].reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']].reset_index(drop=True))
    
    # Save processed data and metadata
    os.makedirs("./processed_data", exist_ok=True)
    train_df.to_csv("./processed_data/train_data.csv", index=False)
    val_df.to_csv("./processed_data/val_data.csv", index=False)
    
    with open("./processed_data/label_mapping.json", "w") as f:
        json.dump(label_mapping, f)
    
    print("Data preparation complete! Files saved to ./processed_data/")
    
    return train_dataset, val_dataset, label_mapping

def load_prepared_data():
    """
    Load previously prepared data
    """
    train_df = pd.read_csv("./processed_data/train_data.csv")
    val_df = pd.read_csv("./processed_data/val_data.csv")
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    with open("./processed_data/label_mapping.json", "r") as f:
        label_mapping = json.load(f)
    
    return train_dataset, val_dataset, label_mapping

if __name__ == "__main__":
    # Run data preparation if executed directly
    train_dataset, val_dataset, label_mapping = prepare_finbert_data()
    print("Data preparation completed successfully!")