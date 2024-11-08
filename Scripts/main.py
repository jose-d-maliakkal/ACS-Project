import pandas as pd
import sys
import os
import json
from sklearn.cluster import KMeans


# Load utils and Scripts directories
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, '..', 'Utils'))
sys.path.append(os.path.join(current_dir, '..', 'Scripts'))



# Load the configuration from config.json
config_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.json')
with open(config_json_path, 'r') as f:
    config = json.load(f)

# Convert relative paths to absolute paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_data_path = os.path.join(project_root, config['raw_data_path'])
cleaned_data_path = os.path.join(project_root, config['cleaned_data_path'])

# Path to save processed datasets
processed_data_path = os.path.join(project_root, 'Data_Preparation', 'Processed_data', 'processed_data.csv')
standard_scaled_data_path = os.path.join(project_root, 'data_preparation', 'scaling_techniques', 'standard_scaled_dataset.csv')
min_max_scaled_data_path = os.path.join(project_root, 'data_preparation', 'scaling_techniques', 'min_max_scaled_dataset.csv')
train_path = os.path.join(project_root, 'Data_Preparation', 'Train_data', 'train_dataset.csv')
test_path = os.path.join(project_root, 'Data_Preparation', 'Test_data', 'test_dataset.csv')
scaled_n_cluster_data_path = os.path.join(project_root, 'Clustering_Analysis', 'Data','scaled_n_cluster_dataset.csv')
visualisations_path = os.path.join(project_root, 'Clustering_Analysis', 'visualisations')


def load_raw_data(file_path):
    """Load raw data from CSV."""
    try:
        df = pd.read_csv(file_path)
        print("Raw data loaded successfully")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        sys.exit()

def clean_data(df):
    """Clean the data using the data_cleaning function."""
    from cleaning import data_cleaning
    cleaned_df = data_cleaning(df)
    if cleaned_df is not None:
        print("Dataset cleaned successfully.")
        return cleaned_df
    else:
        print("Data cleaning failed")
        sys.exit()

def feature_engineering(df):
    """Perform feature engineering on the dataset."""
    from feature_engineering import add_features
    features_df = add_features(df)
    if features_df is not None:
        print("Feature engineering done successfully.")
        return features_df
    else:
        print("Feature engineering failed")
        sys.exit()

def scale_data(df):
    """Scale the dataset using Standard and Min-Max scaling."""
    from scaling_data import  Min_Max_Scaling

    # Min-Max Scaling
    min_max_scaled_df = Min_Max_Scaling(df)
    if min_max_scaled_df is not None:
        print("Min-Max Scaling done successfully")
    else:
        print("Min-Max Scaling failed")
        sys.exit()

    return min_max_scaled_df


def split_data(df, target_column):
    """Split the dataset into training and testing data."""
    from split_data import split_data
    train_df, test_df = split_data(df, target_column)
    if train_df is not None and test_df is not None:
        print("Data successfully split into Train and Test datasets.")
        return train_df, test_df
    else:
        print("Data splitting failed")
        sys.exit()


def perform_clustering(min_max_scaled_df):
    """Apply K-Means clustering and visualize the results."""
    from clustering_analysis import optimise_k_means, apply_k_means, summarize_and_visualize_clusters

    # Optimize K-Means
    data = min_max_scaled_df[['tenure', 'MonthlyCharges']]
    inertias = optimise_k_means(data, max_k=10)
    print("\nThe WCSS (Inertia) values for 1 to 10 clusters:\n", inertias)

    # Apply K-Means with the determined number of clusters
    n_clusters = 4  # 4 is the optimal number from elbow method
    kmeans_instance, scaled_df_n_clusters = apply_k_means(min_max_scaled_df, data, n_clusters, visualisations_path)

    # Save scaled dataset with clusters
    scaled_df_n_clusters.to_csv(scaled_n_cluster_data_path, index=False)

    # Visualize clusters
    summarize_and_visualize_clusters(scaled_df_n_clusters, kmeans_instance, visualisations_path)


def main():
    """Main function for data processing and analysis."""
    # 1. Load the raw data
    print("\n1. Loading the raw data file\n")
    df = load_raw_data(raw_data_path)

    # 2. Clean the data
    print("\n2. Data Cleaning\n")
    df_cleaned = clean_data(df)
    df_cleaned.to_csv(cleaned_data_path, index=False)
    print(f"Saved cleaned data to {cleaned_data_path}")

    # 3. Feature Engineering
    print("\n3. Feature Engineering\n")
    df_features = feature_engineering(df)
    print("Displaying the dataset after feature engineering")
    print(df_features)
    df_features.to_csv(processed_data_path, index=False)
    print(f"Saved dataset after feature engineering to {processed_data_path}")

    # 4. Scaling the data
    print("\n4. Scaling the data set\n")
    min_max_scaled_df = scale_data(df_features)
    print("Displaying the dataset after scaling")
    print(min_max_scaled_df)
    min_max_scaled_df.to_csv(min_max_scaled_data_path, index=False)
    print(f"Saved Min-Max scaled data to {min_max_scaled_data_path}")

    # 5. Split the processed dataset into training and testing data
    print("\n5. Data Splitting \n")
    train_df, test_df = split_data(df_features, target_column='Churn_Yes')
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Print the number of samples and features
    print(f"Train Data: {train_df.shape[0]} samples, {train_df.shape[1]} features")
    print(f"Test Data: {test_df.shape[0]} samples, {test_df.shape[1]} features")
    
    print(f"Train data saved to {train_path}")
    print(f"Test data saved to {test_path}")

    # 6. Clustering Analysis
    print("\n6. Clustering Analysis\n")
    perform_clustering(min_max_scaled_df)

    # 7. Predictive Analysis
    


if __name__ == "__main__":
    main()

