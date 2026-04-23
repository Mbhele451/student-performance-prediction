# Student Performance Prediction Project Documentation

## Project Overview

This project predicts student marks using supervised machine learning. It uses a dataset of student records with three input features:
- `StudyHours`
- `Attendance`
- `SleepHours`

The project trains a Random Forest regression model to estimate final student marks and provides tools for prediction and visualization.

## Repository Files

- `#CSV_Generator.py`
  - Generates synthetic student performance data.
  - Creates `student_performance.csv` with 500 rows of sample data.

- `train_model.py`
  - Loads `student_performance.csv`.
  - Trains a `RandomForestRegressor` on the feature set.
  - Evaluates model performance using MAE and R².
  - Saves the trained model as `student_performance_model.pkl`.

- `predict.py`
  - Loads the trained model from `student_performance_model.pkl`.
  - Predicts marks for a new student record.
  - Computes a confidence range using test set MAE.

- `visualize.py`
  - Loads `student_performance.csv` and the trained model.
  - Produces a model report in `model_report.png`.
  - Includes plots for actual vs predicted values, residuals, feature importance, and study hours vs marks.

- `student_performance.csv`
  - The dataset containing the student features and target marks.

- `student_performance_model.pkl`
  - The serialized trained Random Forest model.

- `model_report.png`
  - Generated visualization report showing model behavior and metrics.

## Installation

### Requirements
- Python 3.7+
- `pandas`
- `scikit-learn`
- `matplotlib`
- `numpy`
- `joblib`
- `reportlab` (for PDF generation)

### Install dependencies

```bash
pip install pandas scikit-learn matplotlib numpy joblib reportlab
```

## How to Use the Code

### 1. Generate Data (optional)

If you want to regenerate the dataset:

```bash
python "#CSV_Generator.py"
```

### 2. Train the Model

Train the Random Forest regression model:

```bash
python train_model.py
```

This script prints:
- dataset shape
- training/test split sizes
- mean absolute error (MAE)
- R² score
- feature importances

It also saves:
- `student_performance_model.pkl`

### 3. Predict Marks for a New Student

Run:

```bash
python predict.py
```

This script loads the saved model and prints:
- predicted marks
- likely prediction range
- grade classification

### 4. Create Visualization Report

Run:

```bash
python visualize.py
```

This script generates `model_report.png` with:
- actual vs predicted marks
- residuals plot
- feature importance chart
- study hours vs marks regression line

## Dataset Description

The dataset contains the following columns:

- `StudyHours`: Hours studied per day (typically 0 to 10)
- `Attendance`: Attendance percentage (40 to 100)
- `SleepHours`: Hours of sleep per night (3 to 10)
- `Marks`: Final score (0 to 100)

## Model Details

### Algorithm
- Random Forest Regression (`sklearn.ensemble.RandomForestRegressor`)

### Input Features
- `StudyHours`
- `Attendance`
- `SleepHours`

### Output
- `Marks`

### Evaluation Metrics
- Mean Absolute Error (MAE)
- R² score

### Feature Importances
The model reports the relative influence of each feature. In the current project, `StudyHours` is typically the most important predictor.

## Example Workflow

```bash
python "#CSV_Generator.py"
python train_model.py
python predict.py
python visualize.py
```

## Notes

- The project uses a fixed random seed (`random_state=42`) for reproducibility.
- Predictions are most reliable for inputs within the dataset range.
- `visualize.py` loads the saved Random Forest model and uses it to create analysis plots.

## PDF Documentation

A PDF version of this documentation is stored as `PROJECT_DOCUMENTATION.pdf`.
