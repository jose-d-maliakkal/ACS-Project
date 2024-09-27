import pandas as pd 
from sklearn.model_selection import train_test_split 

def split_data(df,target_column, test_size=0.2, random_state=42):
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split the data into training and testing sets
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Combine the features and target variables
    train_df = pd.concat([train_X, train_y], axis=1)
    test_df = pd.concat([test_X, test_y], axis=1)
    
    return train_df, test_df
