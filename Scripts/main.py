import pandas as pd
import sys
import os
import json


#Load utils and Scripts directories

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Scripts'))


# Load the configuration from config.json

config_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.json')
with open(config_json_path, 'r') as f:
    config = json.load(f)
        
# Convert relative paths to absolute paths

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_data_path = os.path.join(project_root, config['raw_data_path'])
cleaned_data_path = os.path.join(project_root, config['cleaned_data_path'])

# Path to save the processed dataset that contains features
processed_data_path = os.path.join(project_root, 'Data_Preparation', 'Processed_data', 'processed_data.csv')

#Path to save the scaled Standard and Min_Max dataset
standard_scaled_data_path = os.path.join(project_root, 'data_preparation', 'scaling_techniques', 'standard_scaled_dataset.csv')
min_max_scaled_data_path = os.path.join(project_root, 'data_preparation', 'scaling_techniques', 'min_max_scaled_dataset.csv')

# Path to save the train and test datasets
train_path = os.path.join(project_root, 'Data_Preparation', 'Train_data', 'train_dataset.csv')
test_path = os.path.join(project_root, 'Data_Preparation', 'Test_data', 'test_dataset.csv')



def main():
    
    # 1. Load the raw data

    print("\n1. Loading the raw data file\n")
    raw_data_path = "Data/Raw_data/Dataset (ATS)-1.csv"
    df = pd.read_csv(raw_data_path)
    if df is None:
        print("File not found, exiting the script")
        exit()
    else:
        print(" Raw data loaded successfully")

    #2. Clean the data

    print("\n 2. Data Cleaning\n")
    from clean import data_cleaning
    df_cleaned = data_cleaning(df)
    print("Dataset cleaned successfully, displaying first 5 records \n")
    print(df_cleaned.head(5))
    df_cleaned.to_csv(cleaned_data_path, index=False)
    print(f"Saved cleaned data to {cleaned_data_path}")
    
    #3. Feature Engineering 
     
    print("\n 3. Feature Engineering\n")
    from feature_engineering import add_features
    df_features = add_features(df_cleaned)
    if df_features is None:
        print("Feature Engineering Failed")
        exit()
    else:
        print("Feature Engineering done successfully.")
    print(df_features.head(5))
    df_features.to_csv(processed_data_path, index=False)
    print(f"Saved dataset after feature engineering to {processed_data_path}")
    
    
    # 4. Scaling the data set

    print("\n4. Scaling the data set\n")

    print("\n Standard Scaling")
    from scaling import Standard_Scaling
    std_scaled_df = Standard_Scaling(df_features)
    if std_scaled_df is None:
        print("Standard Scaling failed, exiting...")
        exit()
    else:
        print("Standard Scaling done succcessfully")
        print(std_scaled_df.head(5))
        std_scaled_df.to_csv(standard_scaled_data_path, index=False)
        print(f"Saved standard scaled data to {standard_scaled_data_path}")
        
    print("\nMin_Max  Scaling")
    from scaling import Min_Max_Scaling
    min_max_scaled_df = Min_Max_Scaling(df_features)
    if min_max_scaled_df is None:
        print("Min_Max Scaling failed, exiting...")
        exit()
    else:
        print("Min_Max Scaling done succcessfully")
        print(min_max_scaled_df.head(5))
        min_max_scaled_df.to_csv(min_max_scaled_data_path, index=False)
        print(f"Saved Min-Max scaled data to {min_max_scaled_data_path}")

    # 5. Split the processed dataset into training data and testing data
    print("\n5. Data Splitting \n")
    from split_data import split_data
    train_df, test_df = split_data(df_features, target_column='Churn_Yes')
    if train_df is None or test_df is None:
        print("Data splitting failed")
        exit()
    else:
        print("Data is successfully split into Train data and Test data sets")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        train_num_records = train_df.shape[0]
        test_num_records = test_df.shape[0]
        print(f"Number of records in train dataset = {train_num_records}")
        print(f"Number of records in test dataset = {test_num_records}")
        print(f"Train data saved to {train_path}")
        print(f"Test data saved to {test_path}")
        



if __name__ == "__main__":
    
    main()