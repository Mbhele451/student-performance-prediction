import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

# Generate random data
study = np.round(np.random.uniform(0, 10, n), 1)       # hours studied
attend = np.round(np.random.uniform(40, 100, n), 0)    # attendance %
sleep = np.round(np.random.uniform(3, 10, n), 1)       # sleep hours

# Generate marks based on a pattern + noise
marks = np.clip(
    study * 5.5 + (attend - 70) * 0.4 + (sleep - 6) * 2.5 + np.random.normal(0, 5, n),
    0, 100
)

# Create DataFrame
data = pd.DataFrame({
    'StudyHours': study,
    'Attendance': attend,
    'SleepHours': sleep,
    'Marks': np.round(marks, 1)
})

# Save CSV
data.to_csv('student_performance.csv', index=False)

print("Dataset created successfully")
print(data.head())