import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define the file path to your 50_Startups.csv file
file_path = r"C:\Users\Admin\Desktop\mini\50_Startups.csv"  # Update the path if needed

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Display the content of the DataFrame
x = df.drop(columns=["Profit"])
y = df["Profit"]  # Corrected to access the "Profit" column properly

print("The features are:")
print(x.head(5))

print("\nThe target field is:")
print(y.head(5))

# Define the ColumnTransformer to handle OneHotEncoding for 'State' column
column_transformer = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(drop="first"), ['State'])], 
    remainder="passthrough"
)

# Apply transformations to the feature matrix 'x'
x_data = column_transformer.fit_transform(x)

# Get feature names after transformation
x_dataframe = column_transformer.get_feature_names_out()
x_dataframe = pd.DataFrame(x_data, columns=x_dataframe)  # Convert to DataFrame for better readability

print("\nThe changed dataframe after encoding:")
print(x_dataframe.head(5))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x_data, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestRegressor
n_estimators = 100 
r_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
r_model.fit(X_train, y_train)

# Predict on the test set
y_pred = r_model.predict(X_test)

# Calculate and print performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
test_score = r_model.score(X_test, y_test)

print(f"\nRandom Forest with {n_estimators} estimators:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared Score (R²): {r2}")
print(f"Testing Score (R² on Test Set): {test_score}")
