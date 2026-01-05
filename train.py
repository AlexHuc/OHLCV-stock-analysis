#!/usr/bin/env python
# coding: utf-8


# # 0. Importing the libs and read the data
from pathlib import Path
import warnings
import json

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Test/Train/Val data split
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

# Import necessary for classification model selection
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Import necessary for regression model selection
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV

# For saving the model
import pickle


# # 1. Data preparation and data cleaning

rows = []

for file in Path("data").glob("*.json"):
    with open(file, "r") as f:
        data = json.load(f)
        rows.append(pd.DataFrame(data["ohlcv_data"]))

df = pd.concat(rows, ignore_index=True)

# Clear the UTC from the time column
df["time"] = df["time"].str.split("+").str[0]

# Rename time column to datetime
df.rename(columns={"time": "datetime"}, inplace=True)

# Fix volume
df["volume_fixed"] = df["volume"].combine_first(df["Volume"])

# Drop redundant colums
df = df.drop(columns=["volume", "Volume"])
df = df.rename(columns={"volume_fixed": "volume"})

# Make sure datetime is in datetime format
df['datetime'] = pd.to_datetime(df['datetime'])

# Sort by datetime just in case
df = df.sort_values('datetime').reset_index(drop=True)


# # 2. Feature Engineering

# 1. Return of the previous candle
df['return_1'] = df['close'].pct_change()

# Replace NaN values resulting from pct_change
df['return_1'] = df['return_1'].fillna(0)

# 2. Range of the candle
df['range'] = df['high'] - df['low']

# 3. Body of the candle
df['body'] = (df['close'] - df['open']).abs()

# 4. Rolling volatility (std dev of close over last 5 candles)
df['volatility_5'] = df['close'].rolling(window=5).std()

# 5. Volume change
df['volume_change'] = df['volume'].pct_change()

# Drop all the rows that have infinite values in 'volume_change'
df = df[~df['volume_change'].isin([np.inf, -np.inf])]

# 6. Trend slope over last 5 candles (simple linear regression slope)
def rolling_slope(series, window=5):
    slopes = [np.nan] * (window-1)
    for i in range(window-1, len(series)):
        y = series[i-window+1:i+1].values
        x = np.arange(window)
        # slope = cov(x, y)/var(x)
        slope = np.polyfit(x, y, 1)[0]
        slopes.append(slope)
    return slopes

df['trend_slope_5'] = rolling_slope(df['close'], window=5)

# -------------------------
# Targets
# -------------------------
horizon = 10  # look 10 candles ahead

# 1. Classification target: 1 if price goes up in 10 candles, else 0
df['y_class'] = (df['close'].shift(-horizon) > df['close']).astype(int)

# 2. Regression target: future close price
df['y_reg'] = df['close'].shift(-horizon)

# Remove rows with NaN in targets
df = df.dropna(subset=['y_class', 'y_reg'])

# # 4. Feature and Target Selection

# Classification Target
classification_target = 'y_class'
target_classification = [classification_target]

# Regression Target
regression_target = 'y_reg'
target_regression = [regression_target]

# Classification Features: all features except the target column for classification
classification_features = df.drop(columns=[classification_target, regression_target, 'datetime'])

# Regression Features: all features except the target column for regression
regression_features = df.drop(columns=[regression_target, classification_target, 'datetime'])


# # 5. Split the data - Regression & Classification
# 
# - Split the data in train/val/test sets with 60%/20%/20% distribution.
# ---------------------- CLASSIFICATION SPLIT ----------------------
# Prepare the classification data (features + target)
df_classification = df[classification_features.columns.tolist() + target_classification]

# Split the data into training, validation, and test sets
df_class_full_train, df_class_test = train_test_split(df_classification, test_size=0.2, random_state=1)
df_class_train, df_class_val = train_test_split(df_class_full_train, test_size=0.25, random_state=1)

# Reset indices
df_class_train = df_class_train.reset_index(drop=True)
df_class_val = df_class_val.reset_index(drop=True)
df_class_test = df_class_test.reset_index(drop=True)

# Extract target variables (y_class)
y_class_train = df_class_train[classification_target].values
y_class_val = df_class_val[classification_target].values
y_class_test = df_class_test[classification_target].values

# Drop the target variable from the DataFrames
del df_class_train[classification_target]
del df_class_val[classification_target]
del df_class_test[classification_target]

# Print dataset sizes
print('Training set size:', len(df_class_train), f"{round(len(df_class_train) / (len(df_class_train) + len(df_class_val) + len(df_class_test)), 2)}")
print('Validation set size:', len(df_class_val), f"{round(len(df_class_val) / (len(df_class_train) + len(df_class_val) + len(df_class_test)), 2)}")
print('Test set size:', len(df_class_test), f"{round(len(df_class_test) / (len(df_class_train) + len(df_class_val) + len(df_class_test)), 2)}")

# Convert DataFrames to dictionaries
df_class_train_dicts = df_class_train.to_dict(orient='records')
df_class_val_dicts = df_class_val.to_dict(orient='records')
df_class_test_dicts = df_class_test.to_dict(orient='records')

# Vectorize the data
dv = DictVectorizer(sparse=True)
X_class_train = dv.fit_transform(df_class_train_dicts)
X_class_val = dv.transform(df_class_val_dicts)
X_class_test = dv.transform(df_class_test_dicts)

# ---------------------- REGRESSION SPLIT ----------------------
# Prepare the regression data (features + target)
df_regression = df[regression_features.columns.tolist() + target_regression]

# Split the data into training, validation, and test sets
df_reg_full_train, df_reg_test = train_test_split(df_regression, test_size=0.2, random_state=1)
df_reg_train, df_reg_val = train_test_split(df_reg_full_train, test_size=0.25, random_state=1)

# Reset indices
df_reg_train = df_reg_train.reset_index(drop=True)
df_reg_val = df_reg_val.reset_index(drop=True)
df_reg_test = df_reg_test.reset_index(drop=True)

# Extract target variables (y_reg)
y_reg_train = df_reg_train[regression_target].values
y_reg_val = df_reg_val[regression_target].values
y_reg_test = df_reg_test[regression_target].values

# Drop the target variable from the DataFrames
del df_reg_train[regression_target]
del df_reg_val[regression_target]
del df_reg_test[regression_target]

print('Training set size:', len(df_reg_train), f"{round(len(df_reg_train) / (len(df_reg_train) + len(df_reg_val) + len(df_reg_test)), 2)}")
print('Validation set size:', len(df_reg_val), f"{round(len(df_reg_val) / (len(df_reg_train) + len(df_reg_val) + len(df_reg_test)), 2)}")
print('Test set size:', len(df_reg_test), f"{round(len(df_reg_test) / (len(df_reg_train) + len(df_reg_val) + len(df_reg_test)), 2)}")

# Convert DataFrames to dictionaries
df_reg_train_dicts = df_reg_train.to_dict(orient='records')
df_reg_val_dicts = df_reg_val.to_dict(orient='records')
df_reg_test_dicts = df_reg_test.to_dict(orient='records')

# Vectorize the data
X_reg_train = dv.fit_transform(df_reg_train_dicts)
X_reg_val = dv.transform(df_reg_val_dicts)
X_reg_test = dv.transform(df_reg_test_dicts)


# # 6. Classification Model - XGBClassifier - Hyperparameter Tunning
# Calculate the positive class rate
pos_rate = y_class_train.mean()

# Initialize the XGBoost model
xgb_model_class = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=(1 - pos_rate) / pos_rate,
    random_state=42,
    n_jobs=-1
)

# Define the parameter grid for RandomizedSearchCV
param_grid_class = {
    'n_estimators': [100, 150, 200, 250, 300],
    'learning_rate': [0.001, 0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_lambda': [0, 0.5, 1, 2, 5],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'min_child_weight': [1, 3, 5, 7]
}

# Create the RandomizedSearchCV instance
grid_search_class = RandomizedSearchCV(
    estimator=xgb_model_class,
    param_distributions=param_grid_class,
    n_iter=150,            # Number of random combinations to try
    cv=5,                  # 5-fold cross-validation
    scoring='roc_auc',     # Use ROC-AUC as the evaluation metric
    verbose=3,             # Set verbosity level to see progress
    n_jobs=-1,             # Use all available CPU cores
    random_state=42        # For reproducibility
)

# Fit the RandomizedSearchCV with the training data
print("\n=== TUNING XGBOOST ===")
grid_search_class.fit(X_class_train, y_class_train)

# Print out the best parameters and best score
print(f"\nBest hyperparameters: {grid_search_class.best_params_}")
print(f"Best CV ROC-AUC score: {grid_search_class.best_score_:.4f}")


# # 8. Classification Model - XGBClassifier - Save the Model
class_output_file = f'./models/model_xgb_class_trained.bin'
class_output_file

best_model_class = grid_search_class.best_estimator_
with open(class_output_file, 'wb') as f_out:
    pickle.dump((dv, best_model_class), f_out)


# # 10. Regression Model - XGBRegressor - Hyperparameter Tunning
# Initialize the XGBoost Regressor model
xgb_model_reg = xgb.XGBRegressor(
    objective='reg:squarederror',  # Objective for regression (mean squared error)
    eval_metric='rmse',            # Root Mean Squared Error (RMSE) as evaluation metric
    random_state=42,
    n_jobs=-1
)

# Define the parameter grid for RandomizedSearchCV
param_grid_reg = {
    'n_estimators': [100, 150, 200, 250, 300],
    'learning_rate': [0.001, 0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_lambda': [0, 0.5, 1, 2, 5],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'min_child_weight': [1, 3, 5, 7]
}

# Create the RandomizedSearchCV instance for regression
grid_search_reg = RandomizedSearchCV(
    estimator=xgb_model_reg,
    param_distributions=param_grid_reg,
    n_iter=150,            # Number of random combinations to try
    cv=5,                  # 5-fold cross-validation
    scoring='neg_root_mean_squared_error',  # Use negative RMSE for scoring (since cross_val_score minimizes)
    verbose=3,             # Set verbosity level to see progress
    n_jobs=-1,             # Use all available CPU cores
    random_state=42        # For reproducibility
)

# Fit the RandomizedSearchCV with the training data
print("\n=== TUNING XGBOOST REGRESSION ===")
grid_search_reg.fit(X_reg_train, y_reg_train)

# Print out the best parameters and best score
print(f"\nBest hyperparameters: {grid_search_reg.best_params_}")
print(f"Best CV RMSE score: {-grid_search_reg.best_score_:.4f}") # Negate the score because it was minimized

# # 11. Regression Model - XGBRegressor - Save the Model
reg_output_file = f'./models/model_xgb_reg_trained.bin'
reg_output_file

best_model_reg = grid_search_reg.best_estimator_
with open(reg_output_file, 'wb') as f_out:
    pickle.dump((dv, best_model_reg), f_out)
