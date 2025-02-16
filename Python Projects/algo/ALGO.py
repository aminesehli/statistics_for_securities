import numpy as np
import scipy as sp
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import mplfinance as mpf
import json
import statsmodels.tsa.stattools as ts

stockPeriod = "5y"
stockInterval = "1d"

def fetch_stock_data(ticker, period=stockPeriod, interval=stockInterval, filename="stock_data.csv"):
    """
    Fetches stock data from Yahoo Finance and saves it to a CSV file.
    
    Args:
        ticker (str or list): Stock ticker(s) to fetch.
        period (str): Time period to fetch data for.
        interval (str): Interval of data.
        filename (str): Name of the CSV file to save the data to.
    """
    # Ensure tickers are in list format
    if isinstance(ticker, str):
        ticker = [ticker]

    # Fetch all stock data in one call
    stock_data = yf.download(ticker, period=period, interval=interval, group_by="ticker")
    
    # If fetching multiple tickers, reformat data
    if len(ticker) > 1:
        stock_data = stock_data.stack(future_stack=True, level=0).rename_axis(["Date", "Ticker"]).reset_index()
    else:
        stock_data["Ticker"] = ticker[0]
        stock_data.reset_index(inplace=True)

    # Save to CSV
    stock_data.to_csv(filename, index=False)
    print(f"Stock data saved to {filename}")

    return stock_data

def fetch_closing_prices(tickers, period=stockPeriod, interval=stockInterval, filename="closing_prices.csv"):
    """
    Fetches only the closing prices of stocks in a structured format:
    - Dates as row indices
    - Tickers as column names
    - Data stored as Closing Prices

    Args:
        tickers (list or str): List of stock tickers or a single ticker.
        period (str): Time period (e.g., "6mo", "1y"). Default is "6mo".
        interval (str): Data interval (e.g., "1d", "1wk"). Default is "1d".
        filename (str, optional): CSV filename to save data. If None, data is not saved.

    Returns:
        pd.DataFrame: DataFrame with dates as index and tickers as columns.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    # Efficiently fetch data for multiple tickers at once
    stock_data = yf.download(tickers, period=period, interval=interval)["Close"]

    # Save to CSV if filename is provided
    if filename:
        stock_data.to_csv(filename)
        print(f"Closing prices saved to {filename}")

    return stock_data

def plot_stock_data(df, ticker, ma_window=20):
    """
    Plots stock price data using mplfinance.
    
    Args:
        df (pd.DataFrame): Stock data DataFrame.
        ticker (str): Stock ticker to plot.
    """
    stock_df = df[df["Ticker"] == ticker].copy()
    
    if stock_df.empty:
        print(f"No data found for {ticker}")
        return

    stock_df.index = pd.to_datetime(stock_df.index)

    stock_df[f"MA{ma_window}"] = stock_df["Close"].rolling(window=ma_window).mean()
    movingAvgPlot = [mpf.make_addplot(stock_df[f"MA{ma_window}"], color='orange')]

    # Plot with mplfinance
    mpf.plot(
        stock_df,
        type="candle", 
        style="charles",
        title=f"{ticker} Stock Price",
        ylabel="Price (USD)",
        volume=False,
        show_nontrading=False,
        addplot=movingAvgPlot,
        warn_too_much_data=99999999999999999999999,
        block=False
    )

def print_recent_close(df, ticker):
    """
    Prints the most recent closing price for a given stock.
    
    Args:
        df (pd.DataFrame): Stock data DataFrame.
        ticker (str): Stock ticker.
    """
    stock_df = df[df["Ticker"] == ticker].copy()

    if stock_df.empty:
        print(f"No data found for {ticker}")
        return

    # Ensure 'Date' column is in datetime format and sorted
    if "Date" in stock_df.columns:
        stock_df["Date"] = pd.to_datetime(stock_df["Date"])
        stock_df = stock_df.sort_values(by="Date")

    # Get most recent close
    recent_close = stock_df["Close"].iloc[-1]
    recent_date = stock_df["Date"].iloc[-1]  # Adjusted for reset index

    print(f"{recent_date.date()} | {ticker} closing price: ${recent_close:.2f}")

def plot_correlation_matrix(data, title="Correlation Matrix"):
    """
    Plots a correlation matrix heatmap for the given data.
    
    Args:
        data (pd.DataFrame): Data to plot.
        title (str): Title of the plot.
    """
    corrMatrix = data.corr()
    
    plt.figure(figsize=(8, 8))
    
    sns.heatmap(corrMatrix, annot=True)
    plt.title("Heatmap")

    sns.clustermap(corrMatrix, annot=True, cmap='coolwarm')
    plt.title("Clustermap")

def plot_multiple_stocks(df, tickers):
    """
    Plots the closing prices of multiple stocks on the same line graph.
    
    Args:
        df (pd.DataFrame): DataFrame containing the closing prices of stocks.
        tickers (list): List of stock tickers to plot.
    """
    plt.figure(figsize=(10,6))
    
    # Plot each ticker's closing prices
    for ticker in tickers:
        if ticker in df.columns:
            plt.plot(df.index, df[ticker], label=ticker)
        else:
            print(f"Data for {ticker} not found in the dataset.")
    
    plt.title("Stock Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Closing Price (USD)")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

def plot_stock_spread(df, ticker1, ticker2):
    '''
    Plots the spread between two stocks.
    
    Args:
        df (pd.DataFrame): DataFrame containing the closing prices of stocks.
        ticker1 (str): First stock ticker name.
        ticker2 (str): Second stock ticker name.
        
    Returns:
        spread (pd.Series): Spread between the two stocks.
    '''
    
    plt.figure(figsize=(10,6))

    spread = df[ticker1] - df[ticker2]
    
    plt.plot(spread, label=f"{ticker1} - {ticker2} Spread")
    plt.title("Stock Spread between {} and {}".format(ticker1, ticker2))
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return spread

def fetch_cointegration_value(df, ticker1, ticker2):
    '''
    Fetches the p-value for the augmented Engle-Granger two step cointegration test.
    This determines if the spread is constant over time.
    Null hypothesis: Spread is non-stationary.
    
    Args:
        df (pd.DataFrame): DataFrame containing the closing prices of stocks.
        ticker1 (str): First stock ticker name.
        ticker2 (str): Second stock ticker name.
        
    Returns:
        coint_pval (float): P-value for the cointegration test.
    '''
    result = ts.coint(df[ticker1], df[ticker2])
    coint_pval = result[1]
    print(f"P-value for augmented Engle-Granger two step cointegration: {coint_pval}")
    
    return coint_pval

def fetch_ADF_test(df, ticker1, ticker2):
    '''
    Fetches the p-value for the Augmented Dickey-Fuller test.
    
    Args:
        df (pd.DataFrame): DataFrame containing the closing prices of stocks.
        ticker1 (str): First stock ticker name.
        ticker2 (str): Second stock ticker name.
        
    Returns:
        stock1ADF (tuple): ADF test results for stock 1.
        stock2ADF (tuple): ADF test results for stock 2.
        spreadADF (tuple): ADF test results for the spread.
        ratioADF (tuple): ADF test results for the ratio.
    '''
    stock1ADF = ts.adfuller(df[ticker1])
    stock2ADF = ts.adfuller(df[ticker2])
    spreadADF = ts.adfuller(df[ticker1] - df[ticker2])
    ratioADF = ts.adfuller(df[ticker1] / df[ticker2])
    
    print(f"ADF Test for {ticker1}: {stock1ADF[1]}")
    print(f"ADF Test for {ticker2}: {stock2ADF[1]}")
    print(f"ADF Test for {ticker1} - {ticker2} Spread: {spreadADF[1]}")
    print(f"ADF Test for {ticker1} / {ticker2} Ratio: {ratioADF[1]}")
    
    return stock1ADF, stock2ADF, spreadADF, ratioADF

def plot_price_ratio(df, stock1, stock2):
    '''
    Plots the price ratio between two stocks.
    
    Args:
        df (pd.DataFrame): DataFrame containing the closing prices of stocks.
        stock1 (str): First stock ticker name.
        stock2 (str): Second stock ticker name.
        
    Returns:
        ratio (pd.Series): Price ratio of the two stocks.
    '''
    
    plt.figure(figsize=(10,6))
    ratio = df[stock1] / df[stock2]
    plt.plot(ratio, label=f"{stock1} / {stock2} Ratio")
    plt.axhline(ratio.mean(), color='red', linestyle='--')
    plt.title(f"{stock1} / {stock2} Ratio")
    plt.legend()
    return ratio

def plot_zscore(df, stock1, stock2, ratio):
    '''
    Plots the Z-score of the price ratio between two stocks.
    
    Args:
        df (pd.DataFrame): DataFrame containing the closing prices of stocks.
        stock1 (str): First stock ticker name.
        stock2 (str): Second stock ticker name.
        ratio (pd.Series): Price ratio of the two stocks.
        
    Returns:
        zscore (pd.Series): Z-score of the price ratio.
    '''
    plt.figure(figsize=(10,6))
    zscore = (ratio - ratio.mean()) / ratio.std()
    plt.plot(zscore, label='Z-scores')
    plt.axhline(zscore.mean(), color='black')
    plt.axhline(1.0, color='red', linestyle='--')
    plt.axhline(1.25, color='red', linestyle='--')
    plt.axhline(-1.0, color='green', linestyle='--')
    plt.axhline(-1.25, color='green', linestyle='--')
    plt.title(f"Z-score of ratio of {stock1} and {stock2}")
    plt.legend(loc='best')
    return zscore

stocks = ['NVDA', 'AVGO', 'TSM', 'QCOM', 'INTC', 'AMD', 'ASML', 'ARM', 'TXN', 'AMAT', 'MU', 'ADI', 'SNPS', 'CDNS', 'MRVL', 'NXPI', 'MPWR', 'MCHP', 'KLAC']
dataHist = fetch_stock_data(stocks)
dataClose = fetch_closing_prices(stocks)

print(dataHist.shape)
print(dataHist.columns)
print(dataClose.shape)
print(dataClose.columns)

for stock in stocks:
    print_recent_close(dataHist, stock) 

plot_stock_data(dataHist, 'NXPI')
plot_stock_data(dataHist, 'AMAT')

plot_correlation_matrix(dataClose, title="Stock Closing Prices Correlation Matrix Heatmap")

selected_tickers = ["NXPI", "AMAT"]
filteredData = dataClose[selected_tickers]

plot_multiple_stocks(filteredData, selected_tickers)

spread_basic = plot_stock_spread(dataClose, selected_tickers[0], selected_tickers[1])
priceRatio = plot_price_ratio(dataClose, selected_tickers[0], selected_tickers[1])

pval_coint = fetch_cointegration_value(dataClose, selected_tickers[0], selected_tickers[1])
ADFstock1, ADFstock2, ADFspread, ADFratio = fetch_ADF_test(dataClose, selected_tickers[0], selected_tickers[1])
zscore = plot_zscore(dataClose, selected_tickers[0], selected_tickers[1], priceRatio)

plt.show()