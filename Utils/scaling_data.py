import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def Min_Max_Scaling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scales numeric columns in the DataFrame using Min-Max scaling.

    Args:
        df : The DataFrame containing numeric columns to be scaled.

    Returns:
         A DataFrame with the same structure as the input, 
         where numeric columns have been scaled to the range [0, 1].
    """
    
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    
    # Identify numeric columns to scale
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print(f"Numeric columns identified for scaling: {num_cols.tolist()}")

    # Create a copy of the original DataFrame to avoid modifying it
    df_copy = df.copy()
    
    # Fit and transform the numeric data
    scaled_data = scaler.fit_transform(df_copy[num_cols])
    
    # Create a DataFrame for the scaled data
    scaled_df = pd.DataFrame(scaled_data, columns=num_cols)
    
    # Retain other non-numeric columns in the original DataFrame
    non_numeric_cols = df_copy.drop(columns=num_cols).columns
    scaled_df = pd.concat([scaled_df, df_copy[non_numeric_cols].reset_index(drop=True)], axis=1)

    return scaled_df
