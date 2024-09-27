import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def data_cleaning(df):
    
    # drop rows with missing values
    print("Dropping rows with missing values...")
    initial_shape = df.shape
    df.dropna(inplace=True) 
    print(f"Successfully dropped rows with missing values. "
          f"Rows dropped: {initial_shape[0] - df.shape[0]}")
    
   
    # Remove duplicate values
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    print(f"Successfully dropped rows with duplicate values. "
            f"Rows dropped: {initial_shape[0] - df.shape[0]}")
    
    return df
    
