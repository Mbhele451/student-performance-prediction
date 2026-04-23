# Student Performance Prediction

A machine learning project that predicts student marks based on study hours, attendance, and sleep hours using Random Forest regression.

## Overview

This project builds and trains a predictive model to estimate student performance. The model analyzes the relationship between key factors (study time, class attendance, and sleep duration) and final marks, providing insights into how these variables impact academic success.

## Features

- **Data Generation**: Create synthetic but realistic student performance data
- **Model Training**: Train a Random Forest regression model on student data
- **Predictions**: Predict marks for new students with confidence ranges
- **Visualization**: Comprehensive visualizations including:
  - Actual vs Predicted marks scatter plot
  - Residual analysis
  - Feature importance (coefficient analysis)
  - Study hours vs marks regression line
- **Model Evaluation**: Metrics including Mean Absolute Error (MAE) and R² score

## Project Structure

```
student-performance-prediction/
├── #CSV_Generator.py          # Generates synthetic dataset
├── train_model.py             # Trains the Random Forest regression model
├── predict.py                 # Makes predictions for new students
├── visualize.py               # Creates model analysis visualizations
├── student_performance.csv    # Dataset (500 student records)
├── student_performance_model.pkl  # Trained model (generated after training)
├── model_report.png           # Visualization output
└── README.md                  # This file
```

## Installation

### Requirements
- Python 3.7+
- pandas
- scikit-learn
- matplotlib
- numpy
- joblib

### Setup

1. Clone or download the project
2. Install required packages:
   ```bash
   pip install pandas scikit-learn matplotlib numpy joblib
   ```

## Usage

### 1. Generate Data (Optional)

Create a new synthetic dataset:
```bash
python #CSV_Generator.py
```

This generates 500 student records with study hours, attendance, sleep hours, and marks.

### 2. Train the Model

Train the Random Forest regression model:
```bash
python train_model.py
```

**Output:**
- `student_performance_model.pkl` - The trained model
- Console output showing dataset shape, MAE, R² score, and feature importances

### 3. Make Predictions

Predict marks for a new student:
```bash
python predict.py
```

**Example output:**
```
===================================
  Predicted marks: 43.2 / 100
  Likely range:    32.5 – 53.9
  Grade:           Merit
===================================
```

### 4. Visualize Results

Generate comprehensive model analysis plots:
```bash
python visualize.py
```

**Output:** `model_report.png` containing:
- Actual vs Predicted scatter plot with MAE and R² metrics
- Residuals plot (prediction errors)
- Feature importance bar chart
- Study hours vs marks regression line

## Dataset

The dataset contains 500 student records with the following features:

| Feature | Range | Description |
|---------|-------|-------------|
| StudyHours | 0-10 | Hours studied per day |
| Attendance | 40-100 | Class attendance percentage |
| SleepHours | 3-10 | Hours of sleep per night |
| **Marks** | 0-100 | **Final marks (target variable)** |

## Model Details

### Algorithm
Linear Regression

### Features
- Study Hours
- Attendance
- Sleep Hours

### Performance Metrics
- **Mean Absolute Error (MAE)**: Average prediction error in marks
- **R² Score**: Proportion of variance explained by the model

### Feature Coefficients
Approximately:
- Study Hours: +5.5 marks per additional hour
- Attendance: +0.4 marks per percentage increase
- Sleep Hours: +2.5 marks per additional hour

## Customization

### Modify Predictions
Edit the `new_student` DataFrame in `predict.py`:
```python
new_student = pd.DataFrame({
    "StudyHours": [6.0],      # Change study hours
    "Attendance": [85],        # Change attendance %
    "SleepHours": [7.5]       # Change sleep hours
})
```

### Adjust Train/Test Split
In `train_model.py`, modify the `test_size` parameter:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # Change 0.2 to your desired ratio
)
```

## Output Files

After running the scripts, the following files are generated:

- `student_performance_model.pkl` - Serialized trained model
- `model_report.png` - Comprehensive 4-panel visualization

## Example Workflow

```bash
# 1. Generate synthetic data
python #CSV_Generator.py

# 2. Train the model
python train_model.py

# 3. View predictions
python predict.py

# 4. Visualize results
python visualize.py
```

## Notes

- The model uses a random seed (42) for reproducibility
- The synthetic data is generated to simulate realistic relationships between variables
- The model's predictions work best for inputs within the training data range (0-10 study hours, 40-100 attendance, 3-10 sleep hours)

## License

This project is provided as-is for educational purposes.
