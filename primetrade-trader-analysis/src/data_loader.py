import pandas as pd
import numpy as np
from datetime import datetime

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


# 🔥 MAIN EXECUTION
if __name__ == "__main__":
    fear_greed_path = r"C:\Users\ASUS\Downloads\fear_greed_index.csv"
    trader_data_path = r"C:\Users\ASUS\Downloads\historical_data.csv"

    fear_df = load_fear_greed_data(fear_greed_path)
    trader_df = load_trader_data(trader_data_path)