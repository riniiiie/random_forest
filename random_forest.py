# =====================================
# RANDOM FOREST — SUPERVISED LEARNING
# =====================================

# Step 1: Upload Dataset
from google.colab import files
uploaded = files.upload()

# Step 2: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 3: Load Dataset
df = pd.read_csv(list(uploaded.keys())[0])

# Step 4: Data Preprocessing
if 'Timestamp' in df.columns:
    df = df.drop(columns=['Timestamp'])

# Convert categorical to numeric
df_encoded = pd.get_dummies(df, drop_first=True)

# Target Column
target = 'Rate your academic stress index '

# --- Start of suggested fix ---
# The error "KeyError: "['Academic Stress Index'] not found in axis"" indicates that the column 'Academic Stress Index'
# does not exist in the DataFrame 'df_encoded' after preprocessing.
# This could be due to a typo in the column name or because the column was transformed (e.g., if it was categorical
# and dummified by pd.get_dummies).
# To diagnose, let's print the available columns in df_encoded:
print("Available columns in df_encoded:", df_encoded.columns.tolist())

# Please check the printed list of columns and update the 'target' variable with the correct column name.
# For example, if the target column is named 'Stress Level', you would change:
# target = 'Academic Stress Index'
# to:
# target = 'Stress Level'
# If the target column was categorical and transformed into multiple dummy variables (e.g., 'Stress_High', 'Stress_Medium'),
# you would need a different approach for defining y (e.g., selecting one of the dummy variables for binary classification,
# or handling it as a multi-class problem appropriately).

# For now, we will stop here to let you adjust the target column.
# X = df_encoded.drop(columns=[target]) # This line caused the error
# y = df_encoded[target] # This line caused the error

# If you have identified the correct target column, uncomment the following lines and replace 'Your_Correct_Target_Column'
# with the actual name.
# target = 'Your_Correct_Target_Column'
X = df_encoded.drop(columns=[target])
y = df_encoded[target]
# --- End of suggested fix ---

# Step 5: Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Prediction
y_pred = model.predict(X_test)

# Step 8: Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 9: Feature Importance
importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(10,6))
plt.barh(features, importance)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.show()

Kaggle link-https://www.kaggle.com/datasets/ayeshaimran1619/student-academic-stress-level
