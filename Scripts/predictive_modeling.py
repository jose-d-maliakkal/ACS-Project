import os
import sys
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping  # type: ignore
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve



# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Set up paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config.json')

# Load the configuration file from the main directory
try:
    with open(config_path) as config_file:
        config = json.load(config_file)
    print("Loaded configuration:", config)
except FileNotFoundError:
    print(f"Configuration file not found at {config_path}")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON configuration file: {e}")
    sys.exit(1)


# Check if all necessary keys are in the configuration
required_keys = ['ANN_Model_path', 'scaled_train_data_path']
for key in required_keys:
    if key not in config:
        print(f"Missing key '{key}' in configuration file.")
        sys.exit(1)


ANN_Model_path = os.path.join(base_dir, config['ANN_Model_path'])
scaled_train_data_path = os.path.join(base_dir, config['scaled_train_data_path'])
scaled_test_data_path = os.path.join(base_dir, config['scaled_test_data_path'])
trained_model_path = os.path.join(base_dir, config['trained_model_path'])
results_path = os.path.join(base_dir, config['results_path'])
Visualization_path = os.path.join(base_dir, config['Visualization_path']) 

# Load the pre-scaled dataset
scaled_train_data = pd.read_csv(scaled_train_data_path)

# Determine input dimensions based on provided dataset (excluding target columns)
feature_columns = scaled_train_data.drop(columns=['Churn_No', 'Churn_Yes']).shape[1]
input_dim = feature_columns


# Task 1: Define the ANN model architecture


# Define the ANN model architecture
model = Sequential()

# Input layer and first hidden layer
model.add(Dense(units=64, activation='relu', input_shape=(input_dim,)))  # input_dim is the number of features in your data
model.add(Dropout(0.2))  # Optional dropout layer for regularization

# Second hidden layer
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))  # Optional dropout layer for regularization

# Third hidden layer
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.2))  # Optional dropout layer for regularization

# Output layer (assuming binary classification for churn prediction)
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()


# Define the file path for saving the ANN Model Architecture
architecture_file_path = os.path.join(ANN_Model_path, 'ANN_Model.json')

# check if the ANN Model Architecture file already exists
if os.path.exists(architecture_file_path):
    print(f"Model Architecture file already exists at {architecture_file_path}. Skipping save.")
else:
    with open(architecture_file_path, 'w') as f:
        f.write(model.to_json())
    print(f"Model architecture defined and saved to {architecture_file_path}")



# Task 2: Train the ANN model

# Load the training dataset
scaled_train_data = pd.read_csv(scaled_train_data_path)
scaled_test_data = pd.read_csv(scaled_test_data_path)

# Separate features and target in training data
X_train = scaled_train_data.drop(columns=['Churn_No', 'Churn_Yes']).values  
y_train = scaled_train_data['Churn_Yes'].values

# Separate features and target in testing data
X_test = scaled_test_data.drop(columns=['Churn_No', 'Churn_Yes']).values  
y_test = scaled_test_data['Churn_Yes'].values

# Define ModelCheckpoint to save the best model
checkpoint = ModelCheckpoint(
    filepath=os.path.join(trained_model_path, 'best_model.keras'),      # path to save the model file
    monitor='val_loss',                                                 # monitor validation loss
    save_best_only=True,                                                # save only the best model
    mode='min',                                                         # 'min' because we want to minimize the validation loss
    verbose=1
)

# Define EarlyStopping to stop training when validation loss is not improving
early_stopping = EarlyStopping(
    monitor='val_loss',          # monitor validation loss
    patience=10,                 # number of epochs with no improvement after which training stops
    mode='min',                  # 'min' because we want to minimize the validation loss
    verbose=1,
    restore_best_weights=True    # restores the weights of the best model after stopping
)

# Fit the model with callbacks
history = model.fit(
    X_train, y_train,
    epochs=50,                   # maximum number of epochs
    batch_size=32,               # typical batch size
    validation_split = 0.2,      # Use 20% of the training data for validation
    callbacks=[checkpoint, early_stopping]
)


# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(os.path.join(Visualization_path, 'Model-Accuracy.png.png'))  # Corrected path
plt.close()

print(f"Model Accuracy saved to {os.path.join(Visualization_path, 'Model_Accuracy.png')}")


# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(os.path.join(Visualization_path, 'Model_Loss.png'))  # Corrected path
plt.close()

print(f"Model Loss saved to {os.path.join(Visualization_path, 'Model_Loss.png')}")

# Load the trained model
model = load_model(os.path.join(trained_model_path, 'best_model.keras'))

# Make predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Save the prediction results
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred.flatten(),
    'Prediction_Probability': y_pred_prob.flatten()
})
results_df.to_csv(os.path.join(results_path, 'predictions.csv'), index=False)

print(f"Predictions saved to {os.path.join(results_path, 'predictions.csv')}")

# Generate and save confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(Visualization_path, 'confusion_matrix.png'))  # Corrected path
plt.close()

print(f"Confusion Matrix saved to {os.path.join(Visualization_path, 'confusion_matrix.png')}")

# Classification report
class_report = classification_report(y_test, y_pred)
with open(os.path.join(results_path, 'classification_report.txt'), 'w') as f:
    f.write(class_report)

print(f"Classification report saved to {os.path.join(results_path, 'classification_report.txt')}")

# Generate and save ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig(os.path.join(Visualization_path, 'roc_curve.png'))  # Corrected path
plt.close()

print(f"ROC curve saved to {os.path.join(Visualization_path, 'roc_curve.png')}")

# Generate and save Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(recall, precision, color='green', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig(os.path.join(Visualization_path, 'precision_recall_curve.png'))  # Corrected path
plt.close()

print(f"Precision-Recall curve saved to {os.path.join(Visualization_path, 'precision_recall_curve.png')}")
