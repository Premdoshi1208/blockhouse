import numpy as np
import pandas as pd

def generate_synthetic_data(rows=1000, stocks=['AAPL', 'MSFT', 'GOOGL']):
    """Generate synthetic order book and trade data."""
    np.random.seed(42)
    timestamps = pd.date_range(start='2025-01-01 09:30:00', periods=rows, freq='S')
    data = []
    for stock in stocks:
        for _ in range(rows):
            bid_prices = np.round(np.random.uniform(100, 200, 5), 2)
            ask_prices = bid_prices + np.round(np.random.uniform(0.01, 0.05, 5), 2)
            bid_sizes = np.random.randint(1, 100, 5)
            ask_sizes = np.random.randint(1, 100, 5)
            volume = np.random.randint(100, 1000)
            price_change = np.round(np.random.uniform(-1, 1), 2)

            entry = {
                'timestamp': timestamps[len(data) % rows],
                'stock': stock,
                **{f'bid_price_{i+1}': bid_prices[i] for i in range(5)},
                **{f'ask_price_{i+1}': ask_prices[i] for i in range(5)},
                **{f'bid_size_{i+1}': bid_sizes[i] for i in range(5)},
                **{f'ask_size_{i+1}': ask_sizes[i] for i in range(5)},
                'volume': volume,
                'price_change': price_change
            }
            data.append(entry)
    return pd.DataFrame(data)

def save_synthetic_data():
    """Save synthetic datasets to the data folder."""
    order_book_data = generate_synthetic_data()
    trade_data = order_book_data[['timestamp', 'stock', 'volume', 'price_change']]
    order_book_data.to_csv('data/order_book.csv', index=False)
    trade_data.to_csv('data/trade_data.csv', index=False)

if __name__ == '__main__':
    save_synthetic_data()
