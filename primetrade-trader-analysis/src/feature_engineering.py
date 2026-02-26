import pandas as pd
import numpy as np


# CREATE DAILY METRICS

def create_daily_metrics(trader_df):

    df = trader_df.copy()

    # Ensure Date column
    if 'Date' not in df.columns:
        raise ValueError("❌ 'Date' column missing in trader data")

    df['closedPnL'] = pd.to_numeric(df.get('closedPnL', 0), errors='coerce').fillna(0)
    df['size'] = pd.to_numeric(df.get('size', 0), errors='coerce').fillna(0)
    df['leverage'] = pd.to_numeric(df.get('leverage', 1), errors='coerce').fillna(1)

    # ---------------------------
    # Aggregation
    # ---------------------------
    daily = df.groupby(['Date', 'account']).agg({
        'closedPnL': 'sum',
        'size': ['sum', 'mean', 'count'],
        'leverage': 'mean'
    }).round(2)

    daily.columns = [
        'total_pnl',
        'total_volume',
        'avg_trade_size',
        'trade_count',
        'avg_leverage'
    ]

    daily = daily.reset_index()

    # ---------------------------
    # Win rate
    # ---------------------------
    df['is_winner'] = df['closedPnL'] > 0
    win = df.groupby(['Date', 'account'])['is_winner'].mean() * 100
    daily['win_rate'] = daily.set_index(['Date', 'account']).index.map(win)

    # ---------------------------
    # Long %
    # ---------------------------
    if 'side' in df.columns:
        df['is_long'] = df['side'].astype(str).str.lower() == 'buy'
        long_pct = df.groupby(['Date', 'account'])['is_long'].mean() * 100
        daily['long_pct'] = daily.set_index(['Date', 'account']).index.map(long_pct)
    else:
        daily['long_pct'] = 0

    # ---------------------------
    # Drawdown
    # ---------------------------
    df['pnl_cumsum'] = df.groupby('account')['closedPnL'].cumsum()
    df['running_max'] = df.groupby('account')['pnl_cumsum'].cummax()
    df['drawdown'] = df['pnl_cumsum'] - df['running_max']

    dd = df.groupby(['Date', 'account'])['drawdown'].min()
    daily['max_drawdown'] = daily.set_index(['Date', 'account']).index.map(dd)

    daily = daily.fillna(0)

    return daily


# ===============================
# 🔗 MERGE SENTIMENT
# ===============================
def merge_with_sentiment(metrics, sentiment):

    metrics['Date'] = pd.to_datetime(metrics['Date'], errors='coerce')
    sentiment['Date'] = pd.to_datetime(sentiment['Date'], errors='coerce')

    sentiment = sentiment.drop_duplicates(subset='Date')
    sentiment = sentiment[['Date', 'Classification', 'Sentiment_Score']]

    merged = pd.merge(metrics, sentiment, on='Date', how='left')
    merged = merged.sort_values('Date')

    merged['Classification'] = merged['Classification'].ffill().fillna('Neutral')
    merged['Sentiment_Score'] = merged['Sentiment_Score'].ffill().fillna(50)

    return merged


# ===============================
# ▶ MAIN EXECUTION
# ===============================
from data_loader import load_fear_greed_data, load_trader_data
from data_cleaner import clean_fear_greed, clean_trader_data

if __name__ == "__main__":

    print("🚀 Running Feature Engineering Pipeline...\n")

    # 👉 USE ABSOLUTE PATH (because you already confirmed it works)
    fear_path = r"C:\Users\ASUS\Downloads\fear_greed_index.csv"
    trader_path = r"C:\Users\ASUS\Downloads\historical_data.csv"

    # Load
    fear_df = load_fear_greed_data(fear_path)
    trader_df = load_trader_data(trader_path)

    # Clean
    fear_clean = clean_fear_greed(fear_df)
    trader_clean = clean_trader_data(trader_df)

    print("✅ Cleaning done")

    # Feature engineering
    daily_metrics = create_daily_metrics(trader_clean)
    print("✅ Daily metrics:", daily_metrics.shape)

    final_df = merge_with_sentiment(daily_metrics, fear_clean)
    print("✅ Final dataset:", final_df.shape)

    # Save output
    final_df.to_csv("final_features.csv", index=False)

    print("\n🎯 DONE: final_features.csv saved successfully")