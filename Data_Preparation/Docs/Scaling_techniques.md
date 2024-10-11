# Min-Max Scaling of Data

This document explains the Min-Max scaling technique used for normalizing the numeric columns of  the dataset. Min-Max scaling is a technique that transforms the values of numeric features to a common scale, typically between 0 and 1. This scaling method is particularly useful for algorithms that are sensitive to the magnitude of the data (such as gradient-based methods). Min-Max Scaling will ensure that all features, including the one-hot encoded categorical features, are appropriately scaled to the same range, facilitating effective distance calculations during clustering analysis. 

## What is Min-Max Scaling?

Min-Max scaling, also known as normalization, rescales the data such that the minimum value becomes 0 and the maximum value becomes 1. The formula for Min-Max scaling is:

$X_{\text{scaled}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$

where:
-  X  is the original value,
-  X_min is the minimum value in the dataset,  
-  X_max is the maximum value in the dataset.


This transformation ensures that all numeric features have the same scale, which can improve the performance of machine learning algorithms like k-Means.

## Function Overview

The `Min_Max_Scaling()` function takes the processed dataframe as input and applies Min-Max scaling to all numeric columns.

### Python Code

```python


def Min_Max_Scaling(df: pd.DataFrame) -> pd.DataFrame:
    
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
```

## Example of the dataset after applying the Scaling

| SeniorCitizen | tenure   | MonthlyCharges | gender_Female | ... | Contract_Two year | Churn_No | Churn_Yes | TotalCharges |
|---------------|----------|----------------|---------------|-----|-------------------|----------|-----------|--------------|
| 0.0           | 0.013889 | 0.115423       | 1.0           | ... | 0.0               | 1.0      | 0.0       | 0.003491     |
| 0.0           | 0.472222 | 0.385075       | 0.0           | ... | 0.0               | 1.0      | 0.0       | 0.226468     |
| 0.0           | 0.027778 | 0.354229       | 0.0           | ... | 0.0               | 0.0      | 1.0       | 0.012596     |
| 0.0           | 0.625000 | 0.239303       | 0.0           | ... | 0.0               | 1.0      | 0.0       | 0.222632     |
| 0.0           | 0.027778 | 0.521891       | 1.0           | ... | 0.0               | 0.0      | 1.0       | 0.016000     |


## Conclusion 

Min-Max scaling is a simple yet effective technique for normalizing numerical data, especially when the distribution of the features is not Gaussian. It is ideal for algorithms that are sensitive to the scale of data, such as neural networks. This method can be implemented in Python using the scikit-learn library's MinMaxScaler, as shown in the provided code.