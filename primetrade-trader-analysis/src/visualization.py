import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

sns.set_style("darkgrid")

# ✅ create folders automatically
os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)


# ======================================
# 📊 PnL vs Sentiment
# ======================================
def plot_pnl_by_sentiment(df):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sns.boxplot(data=df, x='Classification', y='total_pnl', ax=axes[0])
    axes[0].set_title('PnL Distribution by Sentiment')
    axes[0].set_yscale('symlog')

    sentiment_stats = df.groupby('Classification')['total_pnl'].agg(['mean', 'sem']).reset_index()
    axes[1].bar(
        sentiment_stats['Classification'],
        sentiment_stats['mean'],
        yerr=1.96 * sentiment_stats['sem'],
        capsize=5
    )
    axes[1].set_title('Average PnL (±95% CI)')

    win_rate = df.groupby('Classification')['win_rate'].mean()
    axes[2].bar(win_rate.index, win_rate.values)
    axes[2].set_title('Average Win Rate by Sentiment')
    axes[2].set_ylabel('Win Rate (%)')

    plt.tight_layout()
    return fig


# ======================================
# 📈 Behavior Changes
# ======================================
def plot_behavior_changes(df):

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sns.barplot(data=df, x='Classification', y='trade_count', ax=axes[0, 0])
    axes[0, 0].set_title('Trade Frequency')

    sns.barplot(data=df, x='Classification', y='avg_leverage', ax=axes[0, 1])
    axes[0, 1].set_title('Average Leverage')

    sns.barplot(data=df, x='Classification', y='avg_trade_size', ax=axes[1, 0])
    axes[1, 0].set_title('Average Trade Size')

    if 'long_pct' in df.columns:
        sns.barplot(data=df, x='Classification', y='long_pct', ax=axes[1, 1])
        axes[1, 1].set_title('Long Position %')

    plt.tight_layout()
    return fig


# ======================================
# 👥 Trader Segmentation
# ======================================
def create_trader_segments(df):

    trader_metrics = df.groupby('account').agg({
        'total_pnl': 'sum',
        'avg_leverage': 'mean',
        'trade_count': 'sum',
        'win_rate': 'mean'
    }).reset_index()

    trader_metrics['leverage_segment'] = pd.cut(
        trader_metrics['avg_leverage'],
        bins=[0, 3, 10, 100],
        labels=['Low (1-3x)', 'Medium (3-10x)', 'High (>10x)']
    )

    trader_metrics['frequency_segment'] = pd.cut(
        trader_metrics['trade_count'],
        bins=[0, 10, 50, float('inf')],
        labels=['Infrequent', 'Regular', 'Frequent']
    )

    trader_metrics['performance_segment'] = pd.cut(
        trader_metrics['win_rate'],
        bins=[0, 40, 60, 100],
        labels=['Low Win Rate', 'Medium Win Rate', 'High Win Rate']
    )

    return trader_metrics


# ======================================
# ▶ MAIN RUN
# ======================================
if __name__ == "__main__":

    df = pd.read_csv("final_features.csv")

    fig1 = plot_pnl_by_sentiment(df)
    fig1.savefig("outputs/charts/pnl_by_sentiment.png")

    fig2 = plot_behavior_changes(df)
    fig2.savefig("outputs/charts/behavior_changes.png")

    segments = create_trader_segments(df)
    segments.to_csv("outputs/tables/trader_segments.csv", index=False)

    print("🎯 Visualization completed and saved in outputs/")