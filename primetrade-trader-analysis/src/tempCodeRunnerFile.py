= load_fear_greed_data("data/raw/fear_greed_index.csv")
    trader_df = load_trader_data("data/raw/hyperliquid_trades.csv")

    clean_fear = clean_fear_greed(fear_df)
    clean_trader = clean_trader_data(trader_df)

    print("✅ Clean Fear:", clean_fear.shape)
    print("✅ Clean Trader:", clean_trader.shape)

    # optional save
    clean_fear.to_csv("data/cleaned_fear_greed.csv", index=False)
    clean_trader.to_csv("data/cleaned_trader_data.csv", index=False)