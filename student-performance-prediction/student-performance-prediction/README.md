# Student Performance Prediction using Machine Learning

An intermediate-level data science project that trains a Random Forest regression model to estimate a student's final exam score from learning-related indicators. It includes an interactive Streamlit web app, a sample CSV dataset, and model evaluation metrics.

> **Important:** The dataset included here is synthetic and created for learning. It is not real student data. Predictions are estimates and should not be used to grade, label, rank, or make high-stakes decisions about students.

## Features

- Predicts an estimated final score from 0 to 100.
- Uses study hours, attendance, previous score, assignment completion, extracurricular participation, and internet access.
- Handles numeric and categorical inputs using a scikit-learn preprocessing pipeline.
- Trains a `RandomForestRegressor`.
- Reports MAE, RMSE, and R² on a held-out test set.
- Displays the sample data and summary statistics in a Streamlit interface.
- Includes validation for required dataset columns.

## Project structure

```text
student-performance-prediction/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── student_performance.csv
```

## Technologies

- Python 3.10+
- Pandas
- scikit-learn
- Streamlit

## Run on Windows 11

1. Install Python from https://www.python.org/downloads/ (Python 3.10 or newer recommended).
2. Extract the project ZIP to a folder.
3. Open Command Prompt in the extracted project folder.
4. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

6. Your browser should open the local Streamlit application. If it does not, follow the local URL printed in the terminal.

## Input columns

| Column | Meaning | Example |
|---|---|---:|
| `study_hours` | Average study hours per day | 3.5 |
| `attendance_percent` | Attendance percentage | 82 |
| `previous_score` | Previous exam score | 68 |
| `assignments_completed` | Assignments completed percentage | 80 |
| `extracurricular` | Participation (`Yes`/`No`) | Yes |
| `internet_access` | Internet available for study (`Yes`/`No`) | Yes |
| `final_score` | Target score, 0–100 | 74 |

## Sample data

The CSV file contains 40 synthetic student examples. A few rows are shown below:

```csv
study_hours,attendance_percent,previous_score,assignments_completed,extracurricular,internet_access,final_score
2.0,70,58,65,No,Yes,59
4.0,85,72,90,Yes,Yes,78
1.5,60,45,50,No,No,48
5.0,92,81,95,Yes,Yes,88
3.0,78,64,75,No,Yes,68
```

The exact complete dataset is available in `data/student_performance.csv`.

## Example app output

The app shows:
- An estimated final score after you enter a student's details.
- A qualitative interpretation such as “Good” or “Needs improvement”.
- Dataset preview and averages.
- MAE, RMSE, R², and actual-vs-predicted test scores.

Metrics can vary if you change the data or model settings. Since this is a small synthetic dataset, evaluation results are for learning the workflow only and are not evidence of real-world accuracy.

## Machine learning workflow

1. Load the CSV data with Pandas.
2. Separate input features and target score.
3. Split the rows into training and test sets.
4. Impute missing values and one-hot encode categorical features.
5. Train a Random Forest regression model.
6. Evaluate on held-out test rows using MAE, RMSE, and R².
7. Retrain the model on all sample rows for the app's prediction form.

## Limitations and ethical use

- Synthetic data does not represent real-world student populations.
- Correlations in a teaching dataset may be artificial or misleading.
- A prediction can be wrong and should not replace teacher judgment or student support.
- Avoid collecting identifiable student information without appropriate permission and safeguards.

## Future improvements

- Compare Linear Regression, Decision Tree, and Random Forest.
- Add cross-validation and feature-importance analysis.
- Improve the dataset using properly consented and anonymized data.
- Add downloadable prediction reports.

## Author

Created as a machine-learning portfolio project.
