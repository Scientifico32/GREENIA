import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def load_data(filepath, sheet_name=None):
    """
    Load data from a CSV or Excel file.
    
    Parameters:
        filepath (str): Path to the file.
        sheet_name (str): Sheet name for Excel files. None for CSV.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        return pd.read_excel(filepath, sheet_name=sheet_name)
    else:
        raise ValueError("Unsupported file format")

def scale_data(scaler, data):
    """
    Scales the data using a given scaler.
    
    Parameters:
        scaler: A fitted Scikit-learn scaler object.
        data (np.array or pd.DataFrame): Data to be scaled.
    
    Returns:
        np.array: Scaled data.
    """
    return scaler.transform(data)

def inverse_scale_data(scaler, data):
    """
    Inverses the scaling of the data using a given scaler.
    
    Parameters:
        scaler: A fitted Scikit-learn scaler object.
        data (np.array or pd.DataFrame): Data to be inverse scaled.
    
    Returns:
        np.array: Inverse scaled data.
    """
    return scaler.inverse_transform(data)

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics: MAE, RMSE, and MAPE.
    
    Parameters:
        y_true (np.array): Ground truth target values.
        y_pred (np.array): Predicted target values.
    
    Returns:
        dict: Dictionary containing MAE, RMSE, and MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def plot_predictions(y_true, y_pred, title='Predictions vs Actuals'):
    """
    Plot the predicted vs actual values.
    
    Parameters:
        y_true (np.array): Ground truth target values.
        y_pred (np.array): Predicted target values.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual Values', marker='o', color='green')
    plt.plot(y_pred, label='Predicted Values', marker='o', color='blue')
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.show()
