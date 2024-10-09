import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds new features to the DataFrame and encodes categorical columns using One-Hot Encoding.

    Args:
        df : The DataFrame to which features will be added.

    Returns:
        The modified DataFrame with new features and encoded categorical columns.
    """

    # Encode categorical columns using One-Hot Encoding
    print("Encoding categorical columns using One-Hot Encoding...")
    categorical_columns = df.select_dtypes(include=['object']).columns
    print(f"Categorical columns identified: {categorical_columns.tolist()}")

    if len(categorical_columns) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = pd.DataFrame(
            encoder.fit_transform(df[categorical_columns]),
            columns=encoder.get_feature_names_out(categorical_columns)
        )

        # Drop the original categorical variables from the DataFrame
        df = df.drop(columns=categorical_columns)
        df.reset_index(drop=True, inplace=True)
        encoded_data.reset_index(drop=True, inplace=True)

        # Concatenate new categorical variables to the DataFrame
        df = pd.concat([df, encoded_data], axis=1)
        print("Categorical columns successfully encoded.")
    else:
        print("No categorical columns found to encode.")

    # Add new features to the dataset
    if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
        df['TotalCharges'] = df['MonthlyCharges'] * df['tenure']
        print("Added new feature - 'TotalCharges' to the dataset.")
    else:
        print("Columns 'MonthlyCharges' and/or 'tenure' are missing.")

    return df
