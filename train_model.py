import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

#training the model using tree-based regression 
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
data = pd.read_csv('student_performance.csv')
print("Dataset loaded successfully")
print("Dataset shape:", data.shape) # Print the shape of the dataset
print(data.head()) # Print the first 5 rows of the dataset

#Feature selection(inputs)
X = data[['StudyHours', 'Attendance', 'SleepHours']] # Features

#Target selection(output)
Y = data['Marks'] # Target variable

# Split the dataset into training and testing sets 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Train the model
print(f"Training set size: {x_train.shape[0]} samples")
print(f"Testing set size: {x_test.shape[0]} samples")

# Initialize and fit the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train) # Learning phase

# Predict the marks for the test set
predictions = model.predict(x_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\nMean Absolute Error: {mae:.2f}")
print(f"R-squared Score: {r2:.2f}")

#Show what the model has learned 
feature_names = ['StudyHours', 'Attendance', 'SleepHours']
for name, imp in zip(feature_names, model.feature_importances_):
    print(f"Feature importance for {name}: {imp:.3f}")

#save the model for future use
joblib.dump(model, 'student_performance_model.pkl')
print("\nModel saved as 'student_performance_model.pkl'")
