# Dataset Splitting and Size Composition

This document describes the process of splitting a dataset into training and testing sets. The dataset is split based on the specified target column, and the size of each set is determined by the `test_size` parameter. The code also ensures the split is reproducible by setting a `random_state`.

## Function Overview

The `split_data` function takes in the following arguments:
- `df`: The input DataFrame containing features and target.
- `target_column`: The name of the target column in the DataFrame, which is the variable we want to predict.
- `test_size`: The proportion of the dataset to include in the test split. The default value is `0.2`, meaning 20% of the data will be used for testing.
- `random_state`: Controls the shuffling of the data before splitting, ensuring reproducibility. The default is `42`.

The function splits the dataset into:
- `train_df`: The training data containing both features and the target.
- `test_df`: The testing data containing both features and the target.

### Code Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df, target_column, test_size=0.2, random_state=42):
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in the DataFrame.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the data into training and testing sets
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Combine the features and target variables
    train_df = pd.concat([train_X, train_y.reset_index(drop=True)], axis=1)
    test_df = pd.concat([test_X, test_y.reset_index(drop=True)], axis=1)

    return train_df, test_df
```

## Size Composition

The `spli_data()` will split the DataFrame df into 80% training data and 20% testing data based on the target column.

After splitting the data composition is:

- **Train Data**: 6657 samples, 19 features
- **Test Data**: 2519 samples, 19 features