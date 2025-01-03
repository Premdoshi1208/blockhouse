import pandas as pd

def preprocess_data(order_book_file, trade_data_file):
    """Load and preprocess order book and trade data."""
    order_book = pd.read_csv(order_book_file)
    trade_data = pd.read_csv(trade_data_file)
    print("OFI Data Columns:", order_book.columns)
    print("Trade Data Columns:", trade_data.columns)
    return order_book, trade_data
