from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.metrics import r2_score

# def cross_impact_analysis(ofi_data, trade_data):
#     """Analyze contemporaneous and lagged cross-impact."""
#     print("OFI Data Columns after calculation:", ofi_data.columns)
#     print("Trade Data Columns before merge:", trade_data.columns)

#     # Merge based on both 'timestamp' and 'stock' columns
#     merged_data = ofi_data.merge(trade_data, on=['timestamp', 'stock'])
#     print("Merged Data Columns:", merged_data.columns)

#     stocks = merged_data['stock'].unique()
#     results = {}
#     for stock in stocks:
#         stock_data = merged_data[merged_data['stock'] == stock]
#         X = stock_data[['OFI_1']]  # Replace with desired OFI level
#         y = stock_data['price_change_y']  # Use 'price_change_y' from trade_data
        
#         # Handling missing values by filling NaNs with the mean of the column
#         imputer = SimpleImputer(strategy='mean')
#         X_imputed = imputer.fit_transform(X)  # Impute missing values in X
#         y_imputed = imputer.fit_transform(y.values.reshape(-1, 1))  # Impute missing values in y
        
#         # Train the model
#         model = LinearRegression()
#         model.fit(X_imputed, y_imputed)
#         y_pred = model.predict(X_imputed)
        
#         results[stock] = {
#             'r2': r2_score(y_imputed, y_pred),
#             'coefficients': model.coef_
#         }
#     return results
# visualization.py
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