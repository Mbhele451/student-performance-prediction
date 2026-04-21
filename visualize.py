'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
data = pd.read_csv('student_performance.csv')

#split the dataset into features and target variable
X = data[["StudyHours", "Attendance", "SleepHours"]]
Y = data["Marks"]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)
# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(Y_test, predictions)
r2 = r2_score(Y_test, predictions)
print(f'Mean Absolute Error: {mae}')
print(f'R^2 Score: {r2}')

# Visualize the results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Student Performance Predictor", fontsize=14)

# Plotting actual vs predicted marks
axes[0].scatter(Y_test, predictions, color='steelblue', alpha=0.6)
axes[0].plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', label='Perfect Prediction')
axes[0].set_xlabel('Actual Marks')
axes[0].set_ylabel('Predicted Marks')
axes[0].set_title('Actual vs Predicted Marks')
axes[0].legend()

# plot study hours vs marks
axes[1].scatter(data['StudyHours'], data['Marks'], color='steelblue', alpha=0.5, label='Students')
xs = np.linspace(0, 10, 100)
ys = model.predict(pd.DataFrame({
    'StudyHours': xs,
    'Attendance': [data['Attendance'].mean()] * 100,
    'SleepHours': [data['SleepHours'].mean()] * 100
}))
# Plot the regression line
axes[1].plot(xs, ys, "r-", linewidth=2, label='Model Line')
axes[1].set_xlabel('Study Hours / Day')
axes[1].set_ylabel('Marks')
axes[1].set_title('Study Hours vs Marks')
axes[1].legend()
# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('student_performance_analysis.png')
plt.show()
print("Analysis complete. Results saved to 'student_performance_analysis.png'.")'''
#another version of visualize.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import joblib

df    = pd.read_csv("student_performance.csv")
X     = df[["StudyHours", "Attendance", "SleepHours"]]
y     = df["Marks"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = joblib.load("student_performance_model.pkl")
preds = model.predict(X_test)
mae   = mean_absolute_error(y_test, preds)
r2    = r2_score(y_test, preds)

fig = plt.figure(figsize=(14, 10))
fig.suptitle("Student Performance Predictor — Model Report", fontsize=15, y=0.98)
gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

# Panel 1: Actual vs Predicted
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test, preds, alpha=0.6, color="steelblue", edgecolors="white", linewidths=0.5)
ax1.plot([20, 100], [20, 100], "r--", linewidth=1.5, label="Perfect prediction")
ax1.set_xlabel("Actual marks")
ax1.set_ylabel("Predicted marks")
ax1.set_title(f"Actual vs Predicted\nMAE={mae:.1f} | R²={r2:.2f}")
ax1.legend(fontsize=9)

# Panel 2: Residuals (prediction errors)
ax2 = fig.add_subplot(gs[0, 1])
residuals = y_test.values - preds
ax2.scatter(preds, residuals, alpha=0.6, color="coral", edgecolors="white", linewidths=0.5)
ax2.axhline(0, color="black", linewidth=1, linestyle="--")
ax2.set_xlabel("Predicted marks")
ax2.set_ylabel("Error (actual − predicted)")
ax2.set_title("Residuals\n(errors should scatter randomly around 0)")

# Panel 3: Feature importance (coefficients)
ax3 = fig.add_subplot(gs[1, 0])
features = ["Study hours", "Attendance", "Sleep hours"]
coefs    = model.coef_
colors   = ["steelblue" if c > 0 else "coral" for c in coefs]
bars     = ax3.barh(features, coefs, color=colors, edgecolor="white")
ax3.axvline(0, color="black", linewidth=0.8)
ax3.set_xlabel("Marks added per unit increase")
ax3.set_title("Feature weights\n(how much each input affects marks)")
for bar, val in zip(bars, coefs):
    ax3.text(val + (0.1 if val >= 0 else -0.1), bar.get_y() + bar.get_height() / 2,
             f"{val:+.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=9)

# Panel 4: Study hours vs marks with model line
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(df["StudyHours"], df["Marks"], alpha=0.4, color="steelblue",
            edgecolors="white", linewidths=0.5, label="Students")
xs = np.linspace(0, 10, 100)
ys = model.predict(pd.DataFrame({
    "StudyHours": xs,
    "Attendance":  [df["Attendance"].mean()] * 100,
    "SleepHours": [df["SleepHours"].mean()] * 100
}))
ax4.plot(xs, ys, "r-", linewidth=2, label="Model line")
ax4.set_xlabel("Study hours / day")
ax4.set_ylabel("Marks")
ax4.set_title("Study hours vs Marks\n(other features held at mean)")
ax4.legend(fontsize=9)

plt.savefig("model_report.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to model_report.png")