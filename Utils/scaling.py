import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def Standard_Scaling(df):
    
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df_copy = df.copy()
    scaled_data = scaler.fit_transform(df_copy[num_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=num_cols) # Fit and tranform numeric data

    return scaled_df

def Min_Max_Scaling(df):
    
    scaler = MinMaxScaler()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df_copy = df.copy()
    scaled_data = scaler.fit_transform(df_copy[num_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=num_cols) # Fit and tranform numeric data
    
    return scaled_df
