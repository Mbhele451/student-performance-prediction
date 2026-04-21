'''import pandas as pd
import joblib

# Load trained model
model = joblib.load("student_performance_model.pkl")

# New student data
new_student = pd.DataFrame({
    'StudyHours': [6.0],
    'Attendance': [85],
    'SleepHours': [7.5]
})

# Predict
predicted = model.predict(new_student)[0]

print(f"Predicted marks: {predicted:.1f}/100")'''
#another version of predict.py
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load model
model = joblib.load("student_performance_model.pkl")

# Reload test set to get MAE for confidence range
df = pd.read_csv("student_performance.csv")
X  = df[["StudyHours", "Attendance", "SleepHours"]]
y  = df["Marks"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mae = mean_absolute_error(y_test, model.predict(X_test))

def get_grade(score):
    if score >= 80: return "Distinction"
    if score >= 65: return "Merit"
    if score >= 50: return "Pass"
    return "At risk"

# New student
new_student = pd.DataFrame({
    "StudyHours": [6.0],
    "Attendance":  [85],
    "SleepHours": [7.5]
})

predicted = model.predict(new_student)[0]
margin    = 1.96 * mae

print("=" * 35)
print(f"  Predicted marks: {predicted:.1f} / 100")
print(f"  Likely range:    {max(0, predicted - margin):.1f} – {min(100, predicted + margin):.1f}")
print(f"  Grade:           {get_grade(predicted)}")
print("=" * 35)