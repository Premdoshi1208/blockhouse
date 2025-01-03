def calculate_ofi(order_book):
    """Compute multi-level OFI metrics for each stock."""
    for level in range(1, 6):
        order_book[f'OFI_{level}'] = (
            (order_book[f'bid_size_{level}'].diff() * (order_book[f'bid_price_{level}'].diff() > 0).astype(int)) -
            (order_book[f'ask_size_{level}'].diff() * (order_book[f'ask_price_{level}'].diff() < 0).astype(int))
        )
    return order_book
