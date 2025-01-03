import matplotlib.pyplot as plt
import seaborn as sns

def plot_ofi_trends(ofi_data, output_path='results/ofi_trends.png'):
    """Plot OFI trends for individual stocks."""
    plt.figure(figsize=(10, 6))
    for stock in ofi_data['stock'].unique():
        stock_data = ofi_data[ofi_data['stock'] == stock]
        plt.plot(stock_data['timestamp'], stock_data['OFI_1'], label=stock)
    plt.legend()
    plt.title('OFI Trends by Stock')
    plt.xlabel('Time')
    plt.ylabel('OFI (Level 1)')
    plt.savefig(output_path)
    plt.show()

def plot_cross_impact_heatmap(cross_impact_matrix, output_path='results/cross_impact_heatmap.png'):
    """Plot heatmap of cross-impact relationships."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cross_impact_matrix, annot=True, cmap='coolwarm')
    plt.title('Cross-Impact Heatmap')
    plt.savefig(output_path)
    plt.show()
