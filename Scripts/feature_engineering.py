import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def add_features(df):
    
    #  Encode categorial columns using oneHot encoding
    
    print(" Encode categorial columns using oneHot encoding")
    categorical_columns = df.select_dtypes(include=['object']).columns
    print(f"Categorical columns identified: {categorical_columns}")
    if len(categorical_columns) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = pd.DataFrame(
                encoder.fit_transform(df[categorical_columns]),
                columns=encoder.get_feature_names_out(categorical_columns)
        )

        df = df.drop(columns=categorical_columns)    # drop the current categorical variables from DF
        df.reset_index(drop=True, inplace=True)
        encoded_data.reset_index(drop=True, inplace=True)
        df = pd.concat([df, encoded_data], axis=1)   # concat new categorical variables to the DF
        print(f"Categorical columns encoded.")
    else:
        print("No categorical columns found to encode")
    
    
    # Add new features to the dataset
    
    if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
        df['TotalCharges'] = df['MonthlyCharges'] * df['tenure']
        print("Added new feature - TotalCharges to the dataset") 
    else:
        print("Columns 'MonthlyCharges' and/or 'tenure' are missing.")
        
    return df
    