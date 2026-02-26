import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
sys.path.append('..')
from src.data_loader import load_fear_greed_data, load_trader_data
from src.feature_engineering import create_daily_metrics, merge_with_sentiment

st.set_page_config(layout="wide", page_title="Trader Performance Dashboard")

st.title("📊 Trader Performance vs Market Sentiment")
st.markdown("Interactive dashboard exploring how trader behavior and performance relate to Fear & Greed index")

# Load data
@st.cache_data
def load_data():
    fear_greed = load_fear_greed_data('../data/raw/fear_greed_index.csv')
    trader_data = load_trader_data('../data/raw/hyperliquid_trades.csv')
    daily_metrics = create_daily_metrics(trader_data)
    final_df = merge_with_sentiment(daily_metrics, fear_greed)
    return final_df

try:
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    selected_sentiment = st.sidebar.multiselect(
        "Select Sentiment",
        options=df['Classification'].unique(),
        default=df['Classification'].unique()
    )
    
    min_date = pd.to_datetime(df['Date']).min()
    max_date = pd.to_datetime(df['Date']).max()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data
    filtered_df = df[
        (df['Classification'].isin(selected_sentiment)) &
        (pd.to_datetime(df['Date']) >= pd.to_datetime(date_range[0])) &
        (pd.to_datetime(df['Date']) <= pd.to_datetime(date_range[1]))
    ]
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total PnL", f"${filtered_df['total_pnl'].sum():,.0f}")
    with col2:
        st.metric("Avg Win Rate", f"{filtered_df['win_rate'].mean():.1f}%")
    with col3:
        st.metric("Avg Leverage", f"{filtered_df['avg_leverage'].mean():.1f}x")
    with col4:
        st.metric("Total Trades", f"{filtered_df['trade_count'].sum():,.0f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(filtered_df, x='Classification', y='total_pnl', 
                     title='PnL Distribution by Sentiment')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        behavior_df = filtered_df.groupby('Classification').agg({
            'trade_count': 'mean',
            'avg_leverage': 'mean',
            'avg_trade_size': 'mean'
        }).reset_index()
        
        fig = px.bar(behavior_df, x='Classification', y=['trade_count', 'avg_leverage'],
                     title='Behavior by Sentiment', barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series
    st.subheader("Performance Over Time")
    daily_agg = filtered_df.groupby('Date').agg({
        'total_pnl': 'sum',
        'Sentiment_Score': 'first'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_agg['Date'], y=daily_agg['total_pnl'].cumsum(),
                             mode='lines', name='Cumulative PnL'))
    fig.add_trace(go.Scatter(x=daily_agg['Date'], y=daily_agg['Sentiment_Score'],
                             mode='lines', name='Sentiment Score', yaxis='y2'))
    fig.update_layout(title='Cumulative PnL vs Sentiment',
                     yaxis=dict(title='Cumulative PnL'),
                     yaxis2=dict(title='Sentiment Score', overlaying='y', side='right'))
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("Raw Data")
    st.dataframe(filtered_df.head(100))

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please ensure data files are in the correct location")