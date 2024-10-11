# WIL Data Analysis Project

## Overview

This project analyzes the data of a telecom company and  delve deep into client customer data, unearth insights, and develop actionable recommendations to mitigate customer churn. This project will illuminate the factors leading to customer churn and pave way to retention strategies to maintain a good customer base. The project includes data preprocessing, clustering analysis, and predictive modeling.

## Directory Structure

```
📁 ACS-PROJECT/
├── 📁 Clustering_Analysis/
│   ├── 📁 Data/
│   ├── 📁 Docs/
│   ├── 📁 visualizations/
├── 📁 Data/
│   ├── 📁 Raw_data/
│   │   ├── 📄 Dataset (ATS)-1.xlsx
│   ├── 📁 Cleaned_data/
│   │   ├── 📄 cleaned_dataset.xlsx
├── 📁 Data_Preparation/
│   ├── 📁 Docs/
│   ├── 📁 Processed_dataset/
│   │   ├── 📄 processed_data.xlsx
│   ├── 📁 Scaling_Techniques/
│   │   ├── 📄 min_max_scaled_dataset.xlsx
│   ├── 📁 Test_data/
│   │   ├── 📄 test_dataset.xlsx
│   ├── 📁 Train_data/
│   │   ├── 📄 train_dataset.xlsx
├── 📁 Predictive_Modeling/
├── 📁 scripts/
│   ├── 📄 clustering_analysis.py
│   ├── 📄 feature_engineering.py
│   ├── 📄 main.py
│   ├── 📄 split_data.py
├── 📁 utils/
│   ├── 📄 clean.py
│   ├── 📄 scaling.py
├── 📁 Video Demo/
├── 📄 .gitignore
├── 📄 config.json
├── 📄 LICENSE
├── 📄 README.md
```

## How to Run the Project

### Setting Up the Environment

1. **Create and Activate a Virtual Environment**:

  - Using `virtualenv`:
    ```bash
    python -m venv environment_name
    source environment_name/bin/activate  
    ```


### Configure Settings

#### **config.json**

Use config.json to ensure paths are correctly set to your project directory structure.


## Running the Scripts

1. **Run the Main Script**:
  ```bash
  python Scripts/main.py
  ```

### Function Descriptions and Program Flow

 1. Load the raw data
    
    ```python 
    load_raw_data(raw_data_path) 
    ```
    This function loads the raw data and return the data to a dataframe.

 2. Clean the data
    ```python 
    clean_data(df)
    ```
    This function cleans the data by dropping the rows with missing values and NULL.
    The cleaned dataset is saved to a dataframe and saved.

 3. Feature Engineering
     ```python 
    feature_engineering(df_cleaned)
     ```
    This function perform feature engineering by using the OneHot encoding method and add a new feature TotalCharges.

 4. Scaling the data
      ```python
    scale_data(df_features)
     ```
    This function perform the scaling of the dataset by using the min_max scaling.

 5. Split the processed dataset into training and testing data
    ```python
    train_df, test_df = split_data(df_features, target_column='Churn_Yes')
     ```
    This function splits the processed dataset into train dataset and test dataset.

 6. Clustering Analysis
    ```python
    perform_clustering(min_max_scaled_df) 
    ```
    By calling this function, we identify the optimal number of cluster using the elbow method and then train the kmean model on this dataset using the identified optimal cluster. The resulting clusters and then visualised based on the cluster characterstics.

## Tasks Completed

### Data Engineering

 1. **Data Loading and Preprocessing**:
     - Cleaned dataset saved to `data/interim/cleaned_dataset.csv`.

 3. **Feature Scaling and Normalization (min_max scaling)**:
      - Scaled dataset saved to `Data_Preparation/Scaling_techniques/`.
      - Dataset with new features saved to `data/processed/processed_data.csv`.

 4. **Data Splitting**:
     - Training datasets saved to `Data_preparation/Train_data/train_data.csv`.
     - Testing  dataset saved to  `Data_preparation/Test_data/test_data.csv` 

### Data Analysis (Clustering Analysis)
 1. **Determining the Optimal Number of Clusters**:
     - Elbow Method plots saved to `Clustering_Analysis/visualisations/elbow_plot.png`.

 2. **Training the Kmeans Model**:
     - Resulting Cluster  saved to `Clustering_Analysis/visualisations/optimal_cluster.png`.

 3. **Visualizing Clusters**:
     - Visualizations on cluster characterstics saved to `Clustering_Analysis/visualizations`.


## Overall Summary

The project successfully completed the data engineering phase and clustering analysis phase of the project. Next phase of the projet is the predictive modelling to determine the customer churn and evaluate the model performance. 