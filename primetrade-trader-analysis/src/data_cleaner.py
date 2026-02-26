import pandas as pd
import numpy as np
from datetime import datetime

# ===========================
# LOAD FUNCTIONS
# ===========================

def load_fear_greed_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Fear & Greed data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    return df


def load_trader_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Trader data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    return df


# ===========================
# CLEAN FUNCTIONS
# ===========================

def clean_fear_greed(df):
    df = df.copy()

    # Convert Date
    if 'date' in df.columns:
        df['Date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date

    # Classification
    if 'classification' in df.columns:
        df['Classification'] = df['classification']
    elif 'Classification' not in df.columns:
        df['Classification'] = 'Neutral'

    df['Classification'] = df['Classification'].fillna('Neutral')

    # Sentiment score
    if 'value' in df.columns:
        df['Sentiment_Score'] = pd.to_numeric(df['value'], errors='coerce').fillna(50)
    else:
        sentiment_map = {'Fear': 20, 'Greed': 80, 'Neutral': 50}
        df['Sentiment_Score'] = df['Classification'].map(sentiment_map).fillna(50)

    return df


def clean_trader_data(df):
    df = df.copy()

    # ===========================
    # 🔥 Timestamp → Date
    # ===========================
    if 'Timestamp IST' in df.columns:
        df['Timestamp IST'] = pd.to_datetime(df['Timestamp IST'], errors='coerce')
        df['Date'] = df['Timestamp IST'].dt.date

    elif 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df['Date'] = df['Timestamp'].dt.date

    else:
        raise ValueError("❌ No Timestamp column found in trader data")

    # ===========================
    # 🔄 Rename columns to match feature_engineering
    # ===========================
    df = df.rename(columns={
        'Account': 'account',
        'Closed PnL': 'closedPnL',
        'Size USD': 'size',
        'Side': 'side'
    })

    # ===========================
    # 🧹 Numeric conversion
    # ===========================
    df['closedPnL'] = pd.to_numeric(df.get('closedPnL', 0), errors='coerce').fillna(0)
    df['size'] = pd.to_numeric(df.get('size', 0), errors='coerce').fillna(0)

    if 'leverage' in df.columns:
        df['leverage'] = pd.to_numeric(df['leverage'], errors='coerce').fillna(1)
    else:
        df['leverage'] = 1

    return df


# ===========================
# ▶ MAIN EXECUTION
# ===========================

if __name__ == "__main__":

    fear_path = r"C:\Users\ASUS\Downloads\fear_greed_index.csv"
    trader_path = r"C:\Users\ASUS\Downloads\historical_data.csv"

    # Load
    fear_df = load_fear_greed_data(fear_path)
    trader_df = load_trader_data(trader_path)

    # Clean
    clean_fear = clean_fear_greed(fear_df)
    clean_trader = clean_trader_data(trader_df)

    # Output
    print("\n✅ Clean Fear:", clean_fear.shape)
    print("✅ Clean Trader:", clean_trader.shape)

    # Save
    clean_fear.to_csv(r"C:\Users\ASUS\Downloads\cleaned_fear_greed.csv", index=False)
    clean_trader.to_csv(r"C:\Users\ASUS\Downloads\cleaned_trader_data.csv", index=False)

    print("\n🎯 Cleaned files saved successfully")