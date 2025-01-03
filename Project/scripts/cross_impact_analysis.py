from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Assuming you have already defined these functions in your project structure

def cross_impact_analysis(ofi_data, trade_data):
    """Analyze contemporaneous and lagged cross-impact."""
    # Merge data
    merged_data = pd.merge(ofi_data, trade_data, on=['timestamp', 'stock'], how='inner')
    
    # Get unique timestamps and stocks
    timestamps = sorted(merged_data['timestamp'].unique())
    stocks = sorted(merged_data['stock'].unique())
    
    # Create empty impact matrix
    impact_matrix = pd.DataFrame(0.0, index=stocks, columns=stocks)
    
    for stock_x in stocks:
        data_x = merged_data[merged_data['stock'] == stock_x]
        
        if len(data_x) == 0:
            continue
            
        X = data_x[['OFI_1']].values
        
        for stock_y in stocks:
            data_y = merged_data[merged_data['stock'] == stock_y]
            
            if len(data_y) == 0:
                continue
                
            # Align data by timestamp
            common_data = pd.merge(
                data_x[['timestamp', 'OFI_1']], 
                data_y[['timestamp', 'price_change_y']], 
                on='timestamp'
            )
            
            if len(common_data) < 2:
                impact_matrix.loc[stock_x, stock_y] = 0.0
                continue
            
            X = common_data[['OFI_1']].values
            y = common_data['price_change_y'].values.reshape(-1, 1)
            
            # Handle missing values
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
            y = imputer.fit_transform(y)
            
            # Fit model
            model = LinearRegression()
            model.fit(X, y)
            
            impact_matrix.loc[stock_x, stock_y] = float(model.coef_[0])
    
    return impact_matrix

def plot_cross_impact_heatmap(matrix, output_path=None):
    """Plot heatmap of cross-impact relationships."""
    if matrix.empty or matrix.ndim != 2:
        raise ValueError("Input matrix is empty or not 2D.")
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Cross-Impact Heatmap')
    if output_path:
        plt.savefig(output_path)
    plt.show()
